import psutil


def show_mem():
    mem = psutil.virtual_memory()
    total = str(round(mem.total / 1024 / 1024))
    # The round method is used to round the number and
    # then convert it into a string. Bytes/1024 is used to get kb and then /1024 is used to get M.
    used = str(round(mem.used / 1024 / 1024))
    use_per = str(round(mem.percent))
    free = str(round(mem.free / 1024 / 1024))
    print("您当前的内存大小为:" + total + "M")
    print("已使用:" + used + "M(" + use_per + "%)")
    print("可用内存:" + free + "M")
