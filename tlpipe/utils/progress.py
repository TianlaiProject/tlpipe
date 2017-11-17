class Progress(object):

    def __init__(self, length, step=None, prog_msg = None):
        self.length = length
        self.step = step
        if not prog_msg is None:
            self.msg = prog_msg
        else:
            self.msg = 'Progress: %d of ' +  ('%d...' % length)

    def show(self, cnt):
        if not self.step is None:
            if cnt % self.step == 0:
                print self.msg % cnt
        else:
            if self.length < 10:
                print self.msg % cnt
            elif self.length < 10000:
                _step, _res = self.length / 10, self.length % 10
                if _res != 0:
                    _step += 1
                if cnt % _step == 0:
                    print self.msg % cnt
            else:
                _step, _res = self.length / 100, self.length % 100
                if _res != 0:
                    _step += 1
                if cnt % _step == 0:
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
