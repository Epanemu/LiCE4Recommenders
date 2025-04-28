#!/bin/sh

# prepare data
python 1_parse_data.py yelp 20 20 50
python 1_parse_data.py netflix 20 20 20
python 1_parse_data.py amazon 50 20 100

# prepare models
for data in yelp netflix amazon ; do
    for fold in 0 1 2 ; do
        python 2_train_models.py $data $fold sum norating
        python 2_train_models.py $data $fold disjunction norating
        python 2_train_models.py $data $fold mean norating
        python 2_train_models.py $data $fold mean rating
    done
done

# compute CEs
for data in yelp netflix amazon ; do
    for fold in 0 1 2 ; do
        for topk in 5 10 ; do
            for drop in nth score ; do
                for lltype in median "0.1" ; do
                    python 3_get_CEs.py $data $fold sum norating $lltype $drop $topk
                    python 3_get_CEs.py $data $fold disjunction norating $lltype $drop $topk
                    python 3_get_CEs.py $data $fold mean norating $lltype $drop $topk
                    python 3_get_CEs.py $data $fold mean rating $lltype $drop $topk
                done
                python 3_get_CEs.py $data $fold mean rating no_spn $drop $topk
                python 3_get_CEs.py $data $fold mean norating no_spn $drop $topk
            done
        done
    done
done