# Train backbone RSIAT-simple cho ket qua cuoi (ghi dump vao exps/simple/dump va exps/multi/dump)
python main.py --config ./exps/simple/imageneta_RSIAT_simple.json
python main.py --config ./exps/simple/imageneta_RSIAT_simple_s1994.json
python main.py --config ./exps/simple/imageneta_RSIAT_simple_s1995.json
python main.py --config ./exps/multi/vtab_simple_s1993.json
python main.py --config ./exps/multi/vtab_simple_s1994.json
python main.py --config ./exps/multi/vtab_simple_s1995.json
# Danh gia offline (khong train): z CA + z LDA + z AGM, lambda = 0.1 tr(G)/M
# python exps/agm/final_eval.py ina_1993=exps/simple/dump/imageneta_RSIAT_simple ina_1994=exps/simple/dump/imageneta_RSIAT_simple_s1994 ina_1995=exps/simple/dump/imageneta_RSIAT_simple_s1995 vtab_1993=exps/multi/dump/vtab_simple_s1993 vtab_1994=exps/multi/dump/vtab_simple_s1994 vtab_1995=exps/multi/dump/vtab_simple_s1995
