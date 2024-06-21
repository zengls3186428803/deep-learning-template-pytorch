import psutil


def show_mem():
    mem = psutil.virtual_memory()
    total = str(round(mem.total / 1024 / 1024))
    # round方法进行四舍五入，然后转换成字符串 字节/1024得到kb 再/1024得到M
    used = str(round(mem.used / 1024 / 1024))
    use_per = str(round(mem.percent))
    free = str(round(mem.free / 1024 / 1024))
    print("您当前的内存大小为:" + total + "M")
    print("已使用:" + used + "M(" + use_per + "%)")
    print("可用内存:" + free + "M")
