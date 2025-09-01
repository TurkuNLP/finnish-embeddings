from config.init_argument_parser import init_argument_parser
from argparse import ArgumentParser, Namespace

def test_initializing_argument_parser():
    parser = init_argument_parser()
    assert isinstance(parser, ArgumentParser)

def test_giving_arguments():

    model_name = "some-awesome-model"

    parser = init_argument_parser()
    args = parser.parse_args([
        model_name,
        "--batch_size", "64"
    ])

    assert args.model_name == model_name
    assert args.batch_size == 64

def test_setting_test_argument():

    model_name = "some-awesome-model"

    parser = init_argument_parser()
    args = parser.parse_args([
        model_name,
        "--test"
    ])

    assert args.test == True

def test_omitting_test_argument():

    model_name = "some-awesome-model"

    parser = init_argument_parser()
    args = parser.parse_args([
        model_name
    ])

    assert args.test == False