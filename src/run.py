from src.necklace import Necklace

dna='''TAAAGCACCCCGCTCGGGTATGGCAGAGAGGACGCCTTCTGAATTGTGCTATCCCTCGACCTTATCAAAGCTTGCTACCAATAATTAGGATTATTGCCTTGCGACAGACCTCCTACTCAGACTGCCTCACATTGAGCTAGTCAGTGAGCGATAAGCTTGACCCGCTTTCTAGGGTCGCGAGTACGTGAACTAGGGCTCCGGACAGGGCTATATACTCGAGTTTGATCTCGCCCCGACAACTGCAAACCTCAACTTTTTTAGATAATATGGTTAGCCGAAGTTGCACGAGGTGCCGTCCGCGGACTGCTCCCCGGGTGTGGCTCCTTCATCTGACAACGTGCAACCCCTATCGCCATCGATTGTTTCTGCGGACGGTGTTGTCCTCATAGTTTGGGCATGTTTCCCTTGTAGGTGTGAAACCACTTAGCTTCGCGCCGTAGTCCTAAAGGAAAACCTATGGACTTTGTTTCGGGTAGCACCAGGAATCTGAACCATGTGAATGTGGACGTGGCGCGCGTACACCTTAATCTCCGGTTCATGCTAGGGATGTGGCTGCATGCTACGTTGACACACCTACACTGCTCGAAGTAAATATACGAAGCGCGCGGCCTGGCCGGAGCCGTTCCGCATCGTCACGTGTTCGTTTACTGTTAATTGGTGGCACATAAGCAATATCGTAGTCCGTCAAATTCAGCCCTGTTATCCCCGGCGTTATGTGTCAAATGGCGTAGAACTGGATTGACTGTTTGACGGTACCTGCTGATCGGTACGGTGACCGAGAATCTGTCGGGCTATGTCACTAATACTTTCCAAACGCCCCGTATCGATGCTGAACGAATCGATGCACGCTCCCGTCTTTGAAAACGCATAAACATACAAGTGGACAGATGATGGGTACGGGCCTCTAATACATCCAACACTCTACGCCCTCTTCAAGAGCTA'''
n=Necklace(dna,3)
from src.localsearch import LocalSearch
l=LocalSearch(t)
a=l.search(10)

a=LocalSearch(n).search(6)

a=LocalSearch(s).search(6)

Necklace(dna,3).get_thief_beads(0)

ssetup = 'from src.necklace import Necklace; test=[3, 0, 0, 0, 2, 1, 0, 1, 1, 1,  3, 0, 2, 3, 1, 1]; n=Necklace(test,3)'
nsetup = 'from src.npnecklace import Necklace; test=[3, 0, 0, 0, 2, 1, 0, 1, 1, 1,  3, 0, 2, 3, 1, 1]; n=Necklace(test,3)'
