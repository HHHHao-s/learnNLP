import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)

tokenizer  = AutoTokenizer.from_pretrained('./model')
model  = AutoModelForCausalLM.from_pretrained('./model')



def inference(lead, keywords=[]):
    """
    lead: 藏头的语句， 比如一个人的名字， 2，3 或4个字
    keywords：关键词, 0~12个关键词比较好
    """
    leading = f"《{lead}》"
    text = "-".join(keywords) + leading
    input_ids = tokenizer(text, return_tensors='pt', ).input_ids[:, :-1]
    lead_tok = tokenizer(lead, return_tensors='pt', ).input_ids[0, 1:-1]

    with torch.no_grad():
        pred = model.generate(
            input_ids,
            max_length=256,
            num_beams=5,
            do_sample=True,
            repetition_penalty=2.1,
            top_p=.6,
            bos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )[0, 1:]

    # 我们需要将[CLS] 字符， 也就是101, 逐个换回藏头的字符
    mask = (pred == 101)
    while mask.sum() < len(lead_tok):
        lead_tok = lead_tok[:mask.sum()]
    while mask.sum() > len(lead_tok):
        reversed_lead_tok = lead_tok.flip(0)
        lead_tok = torch.cat([
            lead_tok, reversed_lead_tok[:mask.sum() - len(lead_tok)]])
    pred[mask] = lead_tok
    # 从 token 编号解码成语句
    generate = tokenizer.decode(pred, skip_special_tokens=True)
    # 清理语句
    generate = generate.replace("》", "》\n").replace("。", "。\n").replace(" ", "")
    return generate


    

inference("林黛玉", ["悲","凄美","才华","弱"])
inference("萧峰", ["豪情","侠","刚","烈"])

input()