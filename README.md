First compile the source code
   $ make

Run it on an example input and get the output
   $ dist/frontier_expansion data/chess.dat data/chess_fim_result_sup_60.txt 60 1 1

if you want to use different number of gpu and cpu,
change the 1 1 to your gpu num and cpu num

60 means to find out the frequent itemset with min support ratio > 60%

The input file format:

Each transaction is stored in one line. Items in the transaction are integers and separated by a space.
For example:

1 2 4 6 7
1 3 4 6
2 5 7
4 6 7 8 9 12
...

The output format is each line have a frequent itemset and its support ratio.
For example

3 (0.888298)

means itemset (3) is frequent and it appears in 88% of the transactions.



