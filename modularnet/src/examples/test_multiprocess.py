import torch
from multiprocessing import Queue
import torch.multiprocessing as mp

def worker(queue, shared_tensor):
    while True:
        rank = queue.get()
        if rank is None:  # Exit signal
            break
        # Each worker will add its rank to the shared tensor
        with shared_tensor.get_lock():  # synchronize access
            shared_data = torch.tensor(shared_tensor, device='cuda')
            shared_data += rank
            print(f'Worker {rank} updated shared tensor to: {shared_data.cpu().numpy()}')

def main():
    # Initialize a shared tensor
    shared_tensor = torch.zeros(1, device='cuda')
    shared_tensor.share_memory_()  # move tensor to shared memory

    # Create a multiprocessing array to share the tensor
    shared_array = mp.Array('d', shared_tensor.cpu().numpy())

    # Create a queue
    queue = Queue()

    # Number of processes
    num_processes = 4

    # Create and start processes
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue, shared_array))
        p.start()
        processes.append(p)

    # Put data into the queue
    for rank in range(num_processes):
        queue.put(rank)

    # Signal the workers to exit
    for _ in range(num_processes):
        queue.put(None)

    # Join processes
    for p in processes:
        p.join()

    # Print final shared tensor
    final_tensor = torch.tensor(shared_array, device='cuda')
    print(f'Final shared tensor: {final_tensor.cpu().numpy()}')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()