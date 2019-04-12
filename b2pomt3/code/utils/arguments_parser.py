from argparse import ArgumentParser


class ArgumentsParser:

    @staticmethod
    def get_parsed_arguments():
        parser = ArgumentParser(
            description="Baseline multitask main file argument parser")
        parser.add_argument("--config",
                            help="File which contains the configuration",
                            default="baseline_input",
                            type=str)

        return vars(parser.parse_args())
