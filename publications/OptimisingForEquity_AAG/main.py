from sensors_urb_obs import main as sensors_urb_obs
from networks_single_obj import main as networks_single_obj
from networks_multi_objs import main as networks_multi_objs
from networks_two_objs import main as networks_two_objs
from figs_demographics import main as figs_demographics
from figs_single_obj import main as figs_single_obj
from figs_urb_obs import main as figs_urb_obs
from figs_multi_objs import main as figs_multi_objs
from figs_two_objs import main as figs_two_objs


def main():
    """
    Run all scripts to process the data, generate optimised networks and save figures
    and a formatted report for a local authority, as defined by the parameters in
    `config.yml`.
    """
    sensors_urb_obs()
    networks_single_obj()
    networks_multi_objs()
    networks_two_objs()
    figs_demographics()
    figs_single_obj()
    figs_urb_obs()
    figs_multi_objs()
    figs_two_objs()


if __name__ == "__main__":
    main()
