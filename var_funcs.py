"""
Definitions of the functions computing all the NN features
"""

import ROOT
import numpy as np

def var_mjj(j1, j2):
    mjj = (j1 + j2).M()
    return mjj

def var_pTjj(j1, j2):
    pTjj = np.sqrt((j1.Px()+j2.Px())**2 + (j1.Py()+j2.Py())**2)
    return pTjj

def var_dEta(j1, j2):
    dEta = j1.Eta() - j2.Eta()
    return dEta

def var_pT_asymm(j1, j2):
    pT_asymm = max(j1.Pt(), j2.Pt()) / (0.5 * (j1.Pt() + j2.Pt()))
    return pT_asymm

def var_rap_prod(j1, j2):
    rap_prod = j1.Rapidity() * j2.Rapidity()
    return rap_prod

def var_jet_props(jetPs, JetbtagUParTAK4QvG, JetbtagUParTAK4B, JetbtagUParTAK4SvUDG, id1, id2):
    props = {}
    for id in [id1, id2]:
        jet_id = jetPs[id]
        props[id] = {
            'pT': jet_id.Pt(),
            'rap': jet_id.Rapidity(),
            'btagUParTAK4QvG': JetbtagUParTAK4QvG[id], 
            'btagUParTAK4B': JetbtagUParTAK4B[id], 
            'JetbtagUParTAK4SvUDG': JetbtagUParTAK4SvUDG[id]
        }
    return props

def var_zepp(pX, j1, j2):
    zepp = (pX.Rapidity() - 0.5 * (j1.Rapidity() + j2.Rapidity())) / (j1.Rapidity() - j2.Rapidity())
    return zepp

def var_pTjj(j1, j2):
    pTjj = np.sqrt((j1.Px()+j2.Px())**2 + (j1.Py()+j2.Py())**2)
    return pTjj

def var_rap_gap(bJets, j1, j2):
    min_rap = min(j1.Rapidity(), j2.Rapidity())
    max_rap = max(j1.Rapidity(), j2.Rapidity())
    rap_gap = sum(min_rap < b_jet.Rapidity() < max_rap for b_jet in bJets)
    return rap_gap

def var_jet_iso(jetPs, id1, id2):
    iso_dict = {
        id1: {'iso': np.inf, 'neighbour_id': None},
        id2: {'iso': np.inf, 'neighbour_id': None}
    }
    for i, jet in enumerate(jetPs):
        for id in iso_dict.keys():
            if i != id:
                diff = jetPs[id].DeltaR(jet, True)
                if min(diff, iso_dict[id]['iso']) == diff:
                    iso_dict[id]['iso'] = diff
                    iso_dict[id]['neighbour_id'] = i
    return iso_dict

def var_ang_seps(bJets, j1, j2):
    jets = {'1': j1, '2': j2}
    ang_seps = {name: [] for name in jets}
    for b_jet in bJets:
        for name, jet in jets.items():
            ang_seps[name].append({
                'DeltaR': jet.DeltaR(b_jet, True),
                'DeltaPhi': jet.DeltaPhi(b_jet),
                'DeltaRap': jet.Rapidity() - b_jet.Rapidity(),
            })
    return ang_seps

def var_cos_theta(pX, j1, j2): # legacy
    c_pX = ROOT.TLorentzVector(pX)
    c_j1 = ROOT.TLorentzVector(j1)
    c_j2 = ROOT.TLorentzVector(j2)
    nVbfPlane = c_j1.Vect().Cross(c_j2.Vect())
    p_p = c_pX.Vect() - nVbfPlane * (c_pX.Vect().Dot(nVbfPlane) / nVbfPlane.Mag2())
    cosTheta= p_p.Mag() / c_pX.Vect().Mag()
    return cosTheta

def var_theta(pX, j1, j2):
    c_pX = ROOT.TLorentzVector(pX)
    c_j1 = ROOT.TLorentzVector(j1)
    c_j2 = ROOT.TLorentzVector(j2)
    pX_vec = c_pX.Vect()
    nVbfPlane = c_j1.Vect().Cross(c_j2.Vect())
    n_hat = nVbfPlane.Unit()
    p_perp = pX_vec.Dot(n_hat)
    p_parallel = pX_vec - n_hat * p_perp
    theta = np.arctan2(p_perp, p_parallel.Mag())
    return theta

def var_pT_balance(pX, j1, j2):
    pT_sum_vect = (pX.Px()+j1.Px()+j2.Px(), pX.Py()+j1.Py()+j2.Py())
    pT_sum_vect = (np.sqrt(pT_sum_vect[0]**2+pT_sum_vect[1]**2))
    pT_sum_scal = (pX.Pt()+j1.Pt()+j2.Pt())
    pT_balance = pT_sum_vect / pT_sum_scal
    return pT_balance