import logging
import argparse
import sys
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")


# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.binary_cross_entropy(input=pred, target=target.float())


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.word_embeddings = nn.Embedding(39, 128)
        self.conv1 = torch.nn.Conv1d(128, 1000, kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(1000*46, 100)
        self.out = nn.Linear(100, 1)


    def forward(self, x):
        embedded = self.word_embeddings(x).permute(0, 2, 1)
        x = F.relu(self.conv1(embedded))
        x = self.dropout(x)
        x = x.view(-1, 1000*46)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.out(x))
        return x


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=30, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )

    args = parser.parse_args(args=args)
    return args


async def fit_model_on_worker(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    #curr_round: int,
    epoch: int,
    max_nr_batches: int,
    lr: float,
):
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        #max_nr_batches=max_nr_batches,
        max_nr_batches=-1,
        epochs=1,
        optimizer="Adam",
        optimizer_args={"lr": lr},
    )
    train_config.send(worker)
    loss = await worker.async_fit(dataset_key="dga", return_ids=[0])
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


def evaluate_model_on_worker(
    model_identifier,
    worker,
    dataset_key,
    model,
    nr_bins,
    batch_size,
    device,
    print_target_hist=False,
):
    model.eval()

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model, loss_fn=loss_fn, optimizer_args=None, epochs=1
    )

    train_config.send(worker)

    result = worker.evaluate(
        dataset_key=dataset_key,
        return_histograms=True,
        nr_bins=nr_bins,
        return_loss=True,
        return_raw_accuracy=True,
        device=device,
    )

    test_loss = result["loss"]
    correct = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_target = result["histogram_target"]
    hist_pred = result["histogram_predictions"]

    if print_target_hist:
        logger.info("Target histogram: %s", hist_target)
    percentage_0 = int(100 * hist_pred[0] / len_dataset)
    percentage_1 = int(100 * hist_pred[1] / len_dataset)
    logger.info(
        "%s: Percentage numbers 0: %s%%, 1: %s%%",
        model_identifier,
        percentage_0,
        percentage_1,
    )

    logger.info(
        "%s: Average loss: %s, Accuracy: %s/%s (%s%%)",
        model_identifier,
        f"{test_loss:.4f}",
        correct,
        len_dataset,
        f"{100.0 * correct / len_dataset:.2f}",
    )


async def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": "127.0.0.1"}
    alice = websocket_client.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
    bob = websocket_client.WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
    charlie = websocket_client.WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)
    testing = websocket_client.WebsocketClientWorker(id="testing", port=8780, **kwargs_websocket)

    for wcw in [alice, bob, charlie, testing]:
        wcw.clear_objects_remote()

    worker_instances = [alice, bob, charlie]

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)

    traced_model = torch.jit.trace(model, torch.zeros([1, 47], dtype=torch.long).to(device))
    learning_rate = args.lr

    for epoch in range(1, 11):
        logger.info("Training epoch %s/%s", epoch, 10)

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    epoch=epoch,
                    max_nr_batches=-1,
                    lr=learning_rate,
                )
                for worker in worker_instances
            ]
        )
        models = {}
        loss_values = {}

        test_models = epoch>0 and epoch<=10
        if test_models:
            logger.info("Evaluating models")
            np.set_printoptions(formatter={"float": "{: .0f}".format})
            for worker_id, worker_model, _ in results:
                evaluate_model_on_worker(
                    model_identifier="Model update " + worker_id,
                    worker=testing,
                    dataset_key="dga_testing",
                    model=worker_model,
                    nr_bins=2,
                    batch_size=500,
                    device=device,
                    print_target_hist=False,
                )

        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        traced_model = utils.federated_avg(models)

        if test_models:
            evaluate_model_on_worker(
                model_identifier="Federated model",
                worker=testing,
                dataset_key="dga_testing",
                model=traced_model,
                nr_bins=2,
                batch_size=500,
                device=device,
                print_target_hist=False,
            )

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())
