import glob
import os
import sys
from typing import Optional


def try_import_energyplus_api(do_import: bool = True):
    """Try to import pyenergyplus, and add the E+ installation to sys.path if it's not found.

    EnergyPlus home can be specified with the ENERGYPLUS_HOME environment variable, in that
    case path is used as is. If ENERGYPLUS_VERSION is specified,
    /usr/local/EnergyPlus-{ENERGYPLUS_VERSION} is used (Linux only). Otherwise, the latest
    E+ installation in /usr/local/EnergyPlus-* is used (Linux only).
    """
    try:
        import pyenergyplus  # noqa
    except ImportError:
        eplus_path = solve_energyplus_install_path()
        assert eplus_path is not None, "Couldn't find any E+ installation"
        assert os.path.exists(eplus_path), f"Couldn't find E+ installation at {eplus_path}"
        sys.path.append(eplus_path)

    try:
        if do_import:
            from pyenergyplus.api import EnergyPlusAPI  # noqa
            from pyenergyplus.datatransfer import DataExchange  # noqa
            from pyenergyplus.runtime import Runtime  # noqa

            return EnergyPlusAPI, DataExchange, Runtime
    except ImportError:
        raise ImportError(
            "Couldn't import pyenergyplus. Please make sure EnergyPlus 9.3 or higher "
            "is correctly installed and available"
        )

    return None, None, None


def solve_energyplus_install_path() -> str:
    eplus_path: Optional[str] = None

    if (eplus_home := os.getenv("ENERGYPLUS_HOME", None)) is not None:
        eplus_path = eplus_home

    elif (eplus_version := os.getenv("ENERGYPLUS_VERSION", None)) is not None:
        eplus_path = f"/usr/local/EnergyPlus-{eplus_version}"

    else:
        eplus_installs = glob.glob("/usr/local/EnergyPlus-*", recursive=False)
        if len(eplus_installs) > 0:
            eplus_path = sorted(eplus_installs, key=lambda x: tuple([int(s) for s in x.split("-")[-3:]]))[-1]

    return eplus_path


def override(cls):
    """Annotation for documenting method overrides.

    :param cls: the superclass that provides the overridden method. If this cls does not
        actually have the method, an error is raised.
    """

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError(f"{method} does not override any method of {cls}")
        return method

    return check_override
