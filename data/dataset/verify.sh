python - <<'PY'
import pickle, pathlib, sys
meta = pickle.load(open('blog_char/meta.pkl','rb'))
print("vocab_size =", meta.get('vocab_size'))
PY