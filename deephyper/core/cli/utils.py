def generate_other_arguments(func, **kwargs):
    cl_format = ""
    for k, v in kwargs.items():
        if v is not None:
            arg = "--" + "-".join(k.split("_"))
            val = str(v)
            arg_val = f"{arg}={val} "
            cl_format += arg_val
    cl_format = cl_format[:-1]
    return cl_format