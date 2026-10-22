"""Rebuild the workbook without rerunning models or starting a tracking server."""
import argparse
from pathlib import Path
from treevibes.tracking import export_leaderboard

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts'))
    parser.add_argument('--workbook',type=Path,default=Path('artifacts/leaderboard.xlsx'))
    args=parser.parse_args()
    print(export_leaderboard(args.output,args.workbook))
