from dataclasses import dataclass
from typing import Callable
from lmfit import  minimize
import pandas as pd
import os

from fit_utils import baseline_data, fit_data, extract_data

BaselineFunction = Callable[[pd.DataFrame, pd.DataFrame, \
                    pd.DataFrame, pd.DataFrame], pd.DataFrame]
FitFunction = Callable[[pd.DataFrame, pd.DataFrame], minimize]
ExtractFunction = Callable[[minimize, minimize, pd.DataFrame], float]

@dataclass
class RatioBot:

    baseline_process: BaselineFunction
    fit_process: FitFunction
    extract_process: ExtractFunction

    def run(self, a: int, b: int, spot: int, line: int, idx: int) -> float:
        print('LOADING DATA')
        d1 = pd.read_csv("col1 line "+str(line) + " Point " + str(spot) + \
            " iteration "+str(idx) + " foreground1D.csv")
        d2 = pd.read_csv("col1 line "+str(line) + " Point " + str(spot) + \
            " iteration "+str(idx) + " foreground2D.csv")
        d1_ = pd.read_csv("../../data/background1D.csv")
        d2_ = pd.read_csv("../../data/background2D.csv")
        
        d1, d2 = self.baseline_process(d1, d2, d1_, d2_)
        out1, out2 = self.fit_process(d1, d2)
        ratio = self.extract_process(out1, out2, d1)
        print(f'GD RATIO: {ratio}')
        return ratio


def main() -> None:
    os.chdir('Campaigns/2022-04-02-5/')
    
    a, b = 1, 2
    spot, line, idx = 1, 0, 1

    bot = RatioBot(baseline_data, fit_data, extract_data)
    ratio = bot.run(a, b, spot, line, idx)
    

if __name__ == '__main__':
    main()

