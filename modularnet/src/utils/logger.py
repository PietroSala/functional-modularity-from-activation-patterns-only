
import os
import wandb


class Logger():
    LEVELS = ['LOG', 'WARNING', 'ERROR']
    
    def __init__(self, base_path, filename='log', backend='file', show_console=True, show_level=None):
        
        self.backend = backend
        self.base_path = base_path
        self.log_file = f"{base_path}/{filename}.txt"
        self.show_console = show_console
        self.show_level = 'LOG' if show_level is None else show_level

        if backend == 'wandb':
            wandb.init(project=base_path, name=filename)
            wandb.login()
        elif backend == 'file':
            os.makedirs(base_path, exist_ok=True)
            with open(self.log_file, 'w') as f: f.write('')

    def log(self, message, level='LOG'):
        msg = self.format_message(message, level )
        self.print_logs(msg, level)

        if self.backend == 'file':
            self.write_file(msg)
        elif self.backend == 'tensorboard':
            print('TensorBoard logging not implemented yet.')
        elif self.backend == 'wandb':
            wandb.log({level: msg})
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def warning(self, message): self.log(message, 'WARNING')
    def error(self, message): self.log(message, 'ERROR')

    def read_logs(self):
        with open(self.log_file, 'r') as f:
            return f.readlines()
        
    def format_message(self, message, level='LOG'):
        return f'[{level}] {message}'
    
    def write_file(self, message):
        with open(self.log_file, 'a') as f: f.write(message+'\n')
    
    def print_logs(self, msg, level='LOG'):
        if not self.show_console: return
        if Logger.LEVELS.index(level) < Logger.LEVELS.index(self.show_level): return
        print(msg)


if __name__ == '__main__':
    logger = Logger(base_path='logs', filename='test_log', backend='file', show_console=True, show_level='LOG')
    logger.log('This is a log message.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')
    
    print("Log file contents:")
    print(''.join(logger.read_logs()))
    logger.log('This is another log message after reading the file.')
    print("Log file contents after writing again:")