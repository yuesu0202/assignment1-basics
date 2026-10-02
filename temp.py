# import chr
import regex as re


def main():
    # chr(0)
    # print(repr(chr(1)))
    # print(chr(1))
    # print(chr(2))
    # print(chr(3))
    # print(chr(4))
    # print(chr(5))
    # print(chr(6))
    # print(chr(7))
    # print(chr(8))
    # print(chr(9))
    # print("this is a test" + chr(0) + "string")
    
    # a = "hello".encode("utf-8")
    # print(a)
    # b = "こんにちは".encode("utf-8")
    # print(b)
    
    special_tokens = ["<|a|>", "<|b|>"]
    special_tokens_str = "|".join(re.escape(special_token) for special_token in special_tokens)
    tokens = re.split(special_tokens_str, "Hello.<|a|>World.<|b|>")
    print(tokens)
if __name__ == "__main__":
    main()
