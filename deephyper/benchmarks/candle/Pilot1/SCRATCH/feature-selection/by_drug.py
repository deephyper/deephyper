from __future__ import print_function

import os
import re

from datasets import NCI60
from argparser import get_parser
from skwrapper import regress, classify, summarize


def test1():
    df = NCI60.load_by_drug_data()
    regress('XGBoost', df)


def test2():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=20)
    df = NCI60.load_by_drug_data()
    regress(model, df, cv=2)


def main():
    description = 'Build ML models to predict by-drug tumor response.'
    parser = get_parser(description)
    args = parser.parse_args()

    print('Args:', args, end='\n\n')
    if args.use_gi50:
        print('Use NCI GI50 value instead of percent growth')
    else:
        print('Use percent growth at log concentration: {}'.format(args.logconc))

    drugs = args.drugs
    if 'all' in drugs:
        drugs = NCI60.all_drugs()
    elif len(drugs) == 1 and re.match("^[ABC]$", drugs[0].upper()):
        drugs = NCI60.drugs_in_set('Jason:' + drugs[0].upper())
        print("Drugs in set '{}': {}".format(args.drugs[0], len(drugs)))

    print()
    for drug in drugs:
        print('-' * 10, 'Drug NSC:', drug, '-' * 10)
        df = NCI60.load_by_drug_data(drug, cell_features=args.cell_features, scaling=args.scaling,
                                     use_gi50=args.use_gi50, logconc=args.logconc,
                                     subsample=args.subsample, feature_subsample=args.feature_subsample)
        if not df.shape[0]:
            print('No response data found\n')
            continue

        if args.classify:
            cutoffs = None if args.autobins > 1 else args.cutoffs
            good_bins = summarize(df, cutoffs, autobins=args.autobins, min_count=args.cv)
            if good_bins < 2:
                print('Not enough classes\n')
                continue
        else:
            summarize(df)

        out = os.path.join(args.out_dir, 'NSC_' + drug)
        for model in args.models:
            if args.classify:
                classify(model, df, cv=args.cv, cutoffs=args.cutoffs, autobins=args.autobins, threads=args.threads, prefix=out)
            else:
                regress(model, df, cv=args.cv, cutoffs=args.cutoffs, threads=args.threads, prefix=out)


if __name__ == '__main__':
    main()
