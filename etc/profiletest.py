import pstats
p = pstats.Stats('results')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats("get",100)