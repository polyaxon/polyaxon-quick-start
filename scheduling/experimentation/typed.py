from polyaxon._k8s.k8s_schemas import V1Container
from polyaxon.schemas import V1IO, V1Component, V1Init, V1Job, V1GitType

"""
This is the same Polyaxonfile as in typed.yaml using the Python library.

Note: Running this file using CLI is similar as well:

```bash
polyaxon run -pm experimentation/typed.py:component -P epochs=5 -l
```

 * -pm: --python-module
"""

inputs = [
    V1IO(name="conv1_size", type="int", value=32, is_optional=True),
    V1IO(name="conv2_size", type="int", value=64, is_optional=True),
    V1IO(name="dropout", type="float", value=0.2, is_optional=True),
    V1IO(name="hidden1_size", type="int", value=500, is_optional=True),
    V1IO(name="conv_activation", type="str", value="relu", is_optional=True),
    V1IO(name="dense_activation", type="str", value="relu", is_optional=True),
    V1IO(name="optimizer", type="str", value="adam", is_optional=True),
    V1IO(name="learning_rate", type="float", value=0.01, is_optional=True),
    V1IO(name="epochs", type="int"),
]

outputs = [
    V1IO(name="loss", type="float"),
    V1IO(name="accuracy", type="float"),
]

job = V1Job(
    init=[
        V1Init(git=V1GitType(url="https://github.com/polyaxon/polyaxon-quick-start"))
    ],
    container=V1Container(
        image="polyaxon/polyaxon-quick-start",
        working_dir="{{ globals.artifacts_path }}",
        command=["python3", "polyaxon-quick-start/model.py"],
        args=[
            "--conv1_size={{ conv1_size }}",
            "--conv2_size={{ conv2_size }}",
            "--dropout={{ dropout }}",
            "--hidden1_size={{ hidden1_size }}",
            "--optimizer={{ optimizer }}",
            "--conv_activation={{ conv_activation }}",
            "--dense_activation={{ dense_activation }}",
            "--learning_rate={{ learning_rate }}",
            "--epochs={{ epochs }}",
        ],
    ),
)

component = V1Component(
    name="typed-experiment",
    description="experiment with inputs",
    tags=["examples"],
    inputs=inputs,
    outputs=outputs,
    run=job,
)
