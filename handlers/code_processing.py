import io
import os
import re
import ast
import json
import sys
import keyword
from fuzzywuzzy import fuzz
from tokenize import tokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER, DEDENT, ERRORTOKEN, NAME
from .python_terminal_command import PythonTerminalCommand
from .testing_util import test_function
dir_path = os.path.dirname(os.path.realpath(__file__))
lit_file = f'{dir_path}/literals.json'

def DecodeIds(idxs, tokenizer, token_types=[]):
    codes = ""
    for idx in idxs:
        to_add = tokenizer.convert_ids_to_tokens(idx)
        if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
            if not codes.endswith(" "):
                codes += " " + to_add[1:]
            else:
                codes += to_add[1:]
        elif (
            idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
            tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT") or to_add in token_types
        ):
            if not codes.endswith(" "):
                codes += " " + to_add + " "
            else:
                codes += to_add + " "
        else:
            codes += to_add
    return codes.strip(" ")
def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens
def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code
def clean_to_code(code_str, post_literal=True):
    code = ""
    if post_literal:
        code_str = post_process(code_str)
    code_str = code_str.replace('<s>', '')
    code_str = code_str.replace('</s>', '')
    code_list = code_str.split()
    indent = 0
    newline = False
    for tok in code_list:
        if '<NUM_LIT:' in tok:
            tok = tok[len('<NUM_LIT:'):-1]
        elif tok == '<NUM_LIT>':
            tok = '0'
        if tok ==  '<INDENT>':
            indent += 1
        elif tok == '<DEDENT>':
            indent -= 1
        elif tok == '<EOL>':
            newline = True
        else:
            if newline:
                code += '\n'
                newline = False
                if indent > 0:
                    code += '\t' * indent
                code += tok
            else:
                code += " " + tok
    return code.strip()

def test_parsable_ast(code_string, return_only_flag=False):
    result = ''
    try:
        ast.parse(code_string)
        result = 'success'
    except Exception as e:
        result = type(e).__name__
    return result=='success' if return_only_flag else (result,result=='success')
def test_compiler(code_string, unique_name='test_compiler'):
    commander = PythonTerminalCommand(unique_name)
    _,_,compilable = commander.process_code_string(code_string, cmd='compiler')
    return compilable
def evaluation(preds_list, gt_list, input_list, eval_list=['em','es','mrr','parsable'], processed=False):
    es = 0.0 ; em = 0.0 ; mrr = 0.0 ; parsable = {'success': 0}
    # accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == test_outputs_text[i]]) / len(predictions)
    for preds, gt, inp in zip(preds_list, gt_list, input_list):
        gt = gt.replace("</s>","").strip()
        pred = preds[0].replace("</s>","").replace("<pad>","").strip()
        em += 1 if 'em' in eval_list and gt == pred else 0
        es += fuzz.ratio(gt, pred) if 'es' in eval_list else 0
        if 'mrr' in eval_list:
            mrr_temp = 0.0
            for mrr_idx in range(len(preds)):
                if preds[mrr_idx].replace("</s>","").replace("<pad>","").strip() == gt and (1/(mrr_idx+1) > mrr_temp):
                    mrr_temp = 1/(mrr_idx+1)
            mrr += mrr_temp
        if 'parsable' in eval_list:
            full_code = inp.replace('<s>', '') + preds[0].replace("</s>","").replace("<pad>","") if not processed else \
                clean_to_code(inp.replace('<s>', '') + ' ' + preds[0].replace("</s>","").replace("<pad>",""))
            result,_ = test_parsable_ast(full_code)
            if result not in parsable:
                parsable[result] = 0
            parsable[result] += 1
    for k in parsable:
        parsable[k] = parsable[k]/len(gt_list)*100
    return em/len(gt_list)*100, es/len(gt_list), mrr/len(gt_list)*100, parsable

def functional_evaluation(preds, solutions, debug=False, single_unittest=False):
    _, errors = test_function(preds, solutions, single_unittest, debug)
    eval_dict = {'assert_correct': 0 }
    for error in errors:
        if error == None:
            eval_dict['assert_correct'] += 1
        else:
            if error not in eval_dict:
                eval_dict[error] = 0
            eval_dict[error] += 1
    for k in eval_dict:
        eval_dict[k] = eval_dict[k]/len(solutions)*100
    return eval_dict

lits = json.load(open(lit_file))
def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits['str']
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )

def preprocess_dataset(code_string, close_tag=True):
    #### extract exact token type from tokenzier library ####
    ## set close_tag=False, if process the unfinish code ##
    transform_dict = {
        NL: "<EOL>",
        NEWLINE: "<EOL>",
        INDENT: "<INDENT>",
        DEDENT: "<DEDENT>",
    }
    out_code = []
    try:
        token_gen = tokenize(io.BytesIO(code_string.encode()).readline)
        was_eol = False
        for tok in token_gen:
            toknum = tok.type
            tokval = " ".join(tok.string.split())
            if toknum == ERRORTOKEN and tokval in [" ",""]:
                continue
            elif toknum in [NEWLINE, NL]:
                if not was_eol:
                    out_code.append("<EOL>")
                    was_eol = True
            elif toknum in transform_dict:
                out_code.append(transform_dict[toknum])
                was_eol = False
            elif toknum == NAME and keyword.iskeyword(tokval):
                out_code.append(tokval)
                was_eol = False
            elif toknum == STRING:
                add_token = process_string(tokval)
                out_code.append(add_token)
                was_eol = False
            elif toknum == NUMBER: 
                if tokval in lits['num']:
                    out_code.append(f"<NUM_LIT:{tokval}>")
                else:
                    out_code.append(f"<NUM_LIT>")
                was_eol = False
            elif toknum not in [COMMENT, ENCODING, ENDMARKER]:
                out_code.append(tokval)
                was_eol = False
        if len(out_code) > 0 and out_code[0] == "<EOL>":
            out_code = out_code[1:]
    except Exception as e:
        print(e)
    if close_tag:
        if len(out_code) > 0 and out_code[0] == "<EOL>":
            out_code.append("<EOL>")
        out_code = ["<s>"] + out_code + ["</s>"]
    else:
        out_code = ["<s>"] + out_code
    out = " ".join(out_code)
    return out
