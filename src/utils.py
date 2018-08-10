"""Some helper functions."""


def convert_string2bool_env(parameter):
    """Convert the String True/False to its boolean form.

    :param parameter: The string that needs to be converted.
    """
    return parameter.lower() == "true"
