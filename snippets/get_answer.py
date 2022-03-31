# colab refused to let me save...

def get_answer(in_model, question):
  source_encoding = tokenizer(
      
      question.get("question"),
      question.get("context"),
      max_length = 396,
      padding = "max_length",
      truncation="only_second",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"


  )

  generated_ids = in_model.model.generate(
      input_ids = source_encoding.get("input_ids"),
      attention_mask = source_encoding.get("attention_mask"),
      num_beams=1,
      max_length=50,
      repetition_penalty=2.45,
      length_penalty=1.0,
      early_stopping=True,
      use_cache=True


  )

  preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
  for gen_id in generated_ids]

  return "".join(preds)
