python -u main.py --lr 0.001 --embedder_lambda 0.1 --optimizer adam --batch_size 40 --epochs 4000 --nonmono 5 --data data/wikitext-2  --dropouth 0.2  --dropouti 0.5 --seed 1882  --adv_lambda 0.02 --log-interval 10 --embedder letter --no-tied --adv --resume WT2_le.pt --save WT2_le.pt 2>&1 | grep -v flatten
