import inspect


def function_arg_checker(function, args_dict):
    def check_option(given_keys, target_keys):
        for given in given_keys:
            if not given in target_keys:
                return False
        return True

    if len(args_dict.keys()) > 0:
        assert check_option(list(args_dict.keys()), list(inspect.getfullargspec(function)[0]),)
