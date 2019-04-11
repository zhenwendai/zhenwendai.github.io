#!/bin/bash

# IP='34.252.149.31'
#
# rsync ubuntu@${IP}:share/experiments/dataEff/regression/naval/*.pdf .
# rsync ubuntu@${IP}:share/experiments/dataEff/regression/power/*.pdf .
# rsync ubuntu@${IP}:share/experiments/dataEff/regression/kin8nm/*.pdf .
# rsync ubuntu@${IP}:share/experiments/dataEff/AL/kin8nm/*.pdf .

pdfcrop --margins 3 braking_all.pdf braking_all.pdf
pdfcrop --margins 3 braking_separate.pdf braking_separate.pdf
pdfcrop --margins 3 braking_lvmogp.pdf braking_lvmogp.pdf
pdfcrop --margins 3 braking_latent_var.pdf braking_latent_var.pdf
pdfcrop --margins 3 servo_data.pdf servo_data.pdf
pdfcrop --margins 3 servo_levelset.pdf servo_levelset.pdf
pdfcrop --margins 3 servo_results.pdf servo_results.pdf
pdfcrop --margins 3 syn_results.pdf syn_results.pdf
pdfcrop --margins 3 syn_md_results.pdf syn_md_results.pdf
pdfcrop --margins 3 syn_md_example.pdf syn_md_example.pdf
pdfcrop --margins 3 sml2010_results.pdf sml2010_results.pdf
