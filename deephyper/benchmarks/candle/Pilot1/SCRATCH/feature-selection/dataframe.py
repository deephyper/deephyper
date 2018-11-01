from __future__ import print_function

import os
import sys

from datasets import NCI60
from argparser import get_parser


description = 'Save dataframes to CSV files.'
parser = get_parser(description)

parser.add_argument("--by", default='drug', choices=['cell', 'drug'],
                    help='generate dataframes for by cell or by drug problems')
parser.add_argument("--float_format", default='%.4g',
                    help='csv float format')

args = parser.parse_args()

sys.stdout = sys.stderr
# print('Args:', args, end='\n\n')

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if args.by == 'cell':
    print('Cell features:', args.cell_features)
    cells = NCI60.all_cells() if 'all' in args.cells else args.cells
    for cell in cells:
        print(cell + ':', end=' ')
        df = NCI60.load_by_cell_data(cell, drug_features=args.drug_features, scaling=args.scaling,
                                     min_logconc=args.min_logconc, max_logconc=args.max_logconc,
                                     subsample=args.subsample, feature_subsample=args.feature_subsample,
                                     verbose=False)
        if df.shape[0]:
            fname = os.path.join(args.out_dir, cell + '.csv')
            df.to_csv(fname, float_format=args.float_format)
            print('saved {} rows and {} columns to {}'.format(df.shape[0], df.shape[1], os.path.normpath(fname)))
        else:
            print('no response data found')
            continue
else:
    print('Drug features:', args.drug_features)
    drugs = args.drugs
    for drug in drugs:
        print('NSC ' + drug + ':', end=' ')
        df = NCI60.load_by_drug_data(drug, cell_features=args.cell_features, scaling=args.scaling,
                                     use_gi50=args.use_gi50, logconc=args.logconc,
                                     subsample=args.subsample, feature_subsample=args.feature_subsample,
                                     verbose=False)
        if df.shape[0]:
            fname = os.path.join(args.out_dir, 'NSC_' + drug + '.csv')
            df.to_csv(fname, float_format=args.float_format)
            print('saved {} rows and {} columns to {}'.format(df.shape[0], df.shape[1], os.path.normpath(fname)))
        else:
            print('no response data found')
            continue
