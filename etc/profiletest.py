#Use this script to analyze the results from using python profile
import pstats
p = pstats.Stats('results')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats("get",100)