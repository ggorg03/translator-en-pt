def parse_arguments(args):
    arguments = {}
    
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=")
            arguments[key] = value
        else:
            print(f"Argumento inválido: {arg}")

    if 'epochs' in arguments.keys():
        arguments['epochs'] = int(arguments['epochs'])
    if 'data_frac' in arguments.keys():
        arguments['data_frac'] = float(arguments['data_frac'])
    if 'checkpoint_path' in arguments.keys():
        arguments['checkpoint_path'] = arguments['checkpoint_path']

    return arguments