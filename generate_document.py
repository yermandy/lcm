import os

import dominate
from dominate.tags import *
from os.path import exists


plots_folder = 'results/plots'

doc = dominate.document(title='Summary')

with doc.head:
    link(rel='stylesheet', href='style.css')
    script(type='text/javascript', src='script.js')

with doc:
    attr(style='font-family:verdana')
    with div():
        with div(style='text-align:center'):
            if exists(f'{plots_folder}/age_density.png'):
                img(src='age_density.png', style='width: 49%')
            if exists(f'{plots_folder}/age_mae.png'):
                img(src='age_mae.png', style='width: 49%')
            hr()
        with div(style='text-align:center'):
            p('μ(a,g)')
            img(src='mu.png', style='width: 75%')
            hr()
        with div(style='text-align:center'):
            p('σ(a,g)')
            img(src='sigma.png', style='width: 75%')
            hr()
        with div(style='text-align:center'):
            p('P(ĝ|a,g)')
            img(src='gamma.png', style='width: 75%')
        
        
with open(f'{plots_folder}/summary.html', 'w') as file:
    file.write(str(doc))