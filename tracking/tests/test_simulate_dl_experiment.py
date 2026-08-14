import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch


def load_simulator(is_managed):
    settings = types.SimpleNamespace(
        CLIENT_CONFIG=types.SimpleNamespace(is_managed=is_managed)
    )
    tracking = types.SimpleNamespace(init=Mock())
    polyaxon = types.ModuleType("polyaxon")
    polyaxon.settings = settings
    polyaxon.tracking = tracking

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = Mock()
    pyplot = types.ModuleType("matplotlib.pyplot")
    numpy = types.ModuleType("numpy")

    module_path = Path(__file__).parents[1] / "simulate_dl_experiment.py"
    spec = importlib.util.spec_from_file_location("simulate_dl_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "matplotlib": matplotlib,
            "matplotlib.pyplot": pyplot,
            "numpy": numpy,
            "polyaxon": polyaxon,
        },
    ):
        spec.loader.exec_module(module)
    return module, tracking


class InitTrackingTest(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(project="quick-start", run_name="baseline")

    def test_managed_run_uses_injected_context(self):
        simulator, tracking = load_simulator(is_managed=True)

        run = simulator.init_tracking(self.args)

        tracking.init.assert_called_once_with()
        self.assertIs(run, tracking.init.return_value)

    def test_local_run_uses_script_arguments(self):
        simulator, tracking = load_simulator(is_managed=False)

        run = simulator.init_tracking(self.args)

        tracking.init.assert_called_once_with(
            project="quick-start",
            name="baseline",
            tags=["sim"],
        )
        self.assertIs(run, tracking.init.return_value)


class MatplotlibBackendTest(unittest.TestCase):
    def test_uses_headless_backend(self):
        simulator, _ = load_simulator(is_managed=False)

        simulator.matplotlib.use.assert_called_once_with("Agg")


class TrapezoidalAreaTest(unittest.TestCase):
    def test_uses_current_numpy_api(self):
        simulator, _ = load_simulator(is_managed=False)
        simulator.np.trapezoid = Mock(return_value=0.75)

        area = simulator.trapezoidal_area([0, 1], [0, 1])

        simulator.np.trapezoid.assert_called_once_with([0, 1], [0, 1])
        self.assertEqual(area, 0.75)

    def test_supports_older_numpy(self):
        simulator, _ = load_simulator(is_managed=False)
        simulator.np.trapz = Mock(return_value=0.75)

        area = simulator.trapezoidal_area([0, 1], [0, 1])

        simulator.np.trapz.assert_called_once_with([0, 1], [0, 1])
        self.assertEqual(area, 0.75)


if __name__ == "__main__":
    unittest.main()
