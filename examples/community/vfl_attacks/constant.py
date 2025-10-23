
def init():
    global values
    values = {'file_name':None}

def set_value(name, value):
    values[name] = value

def get_value(name):
    if name in values.keys():
        return values[name]
    else:
        return None
