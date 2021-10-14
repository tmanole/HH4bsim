import ROOT

def get_norm(bbbj, bbbb):

    w3b_CR_sum = 0.0
    w3b_SR_sum = 0.0
    w4b_CR_sum = 0.0

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.CR == 1:
            w3b_CR_sum += bbbj.weight

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1:
            w3b_SR_sum += bbbj.weight

    for i in range(bbbb.GetEntries()):
        bbbb.GetEntry(i)

        if bbbb.CR == 1:
            w4b_CR_sum += bbbb.weight

    return w3b_CR_sum, w3b_SR_sum, w4b_CR_sum
    



