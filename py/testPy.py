

def func():
    func.a = 111
    print(func.a)

def func_a():
    print(func.a)

func()
func_a()