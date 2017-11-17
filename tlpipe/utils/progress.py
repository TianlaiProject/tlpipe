class Progress(object):

    def __init__(self, length, step=None, prog_msg = None):
        if not prog_msg is None:
            self.msg = prog_msg
        else:
            self.msg = 'Progress: %d of ' +  ('%d...' % length)

        if step is None:
            if length < 10:
                step = 1
            elif length < 10000:
                step = length / 10
            else:
                step = length / 100

        if length <= 0:
            self.cnts = []
        else:
            num = length / step + 1
            self.cnts = [ i * step for i in range(num) ]

    def show(self, cnt):
        if cnt in self.cnts:
            print self.msg % cnt


if __name__ == '__main__':
    lg = 7
    pg = Progress(lg, step=2)
    for i in range(lg):
        pg.show(i)

    pg1 = Progress(lg, step=None)
    for i in range(lg):
        pg1.show(i)

    lg = 20
    pg2 = Progress(lg, step=None)
    for i in range(lg):
        pg2.show(i)

    lg = 23
    pg3 = Progress(lg, step=None)
    for i in range(lg):
        pg3.show(i)

    lg = 10106
    pg4 = Progress(lg, step=None)
    for i in range(lg):
        pg4.show(i)
