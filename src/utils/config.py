import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--strategy',
        type=str,
        default='Replay'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01, 
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0
    )
    parser.add_argument(
        '--l2',
        type=float,
        dest='weight_decay',
        default=0.01,
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
    )
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
        default=1
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=200,  
    )
    parser.add_argument(
        '--memory-size',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--interactive',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--save_folder_name',
        type=str,
        default='.'
    )
    
    return parser.parse_args()