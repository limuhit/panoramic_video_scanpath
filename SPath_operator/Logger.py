class Logger():
    
    def __init__(self, fname, screen = True, file = True):
        self.file = file
        self.fout = open(fname, 'w') if file else None
        self.screen_out = screen
    
    def log(self, *args):
        message = " ".join(str(arg) for arg in args)
        if self.screen_out:
            print(message)
        if self.file:
            self.fout.write(message)
            self.fout.write("\n")
            self.fout.flush()
    
    def close(self):
        if self.fout is not None:
            self.fout.close()
            self.fout = None

    def __del__(self):
        self.close()
