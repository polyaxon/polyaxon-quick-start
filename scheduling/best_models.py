import argparse

from polyaxon.client import RunClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str)
    parser.add_argument("--top", type=int)
    args = parser.parse_args()

    project = args.project

    client = RunClient()

    print("Top 5 experiment based on accuracy for project {}: ".format(project))
    for run in client.list(
        query="metrics.accuracy:>0.9, project.name:{}".format(project),
        sort="-metrics.accuracy",
        limit=args.top,
    ).results:
        print("Run", run.uuid)
        print("Inputs", run.inputs)
        print("Outputs", run.outputs)
