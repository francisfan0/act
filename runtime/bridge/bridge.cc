// Thin actsim ABI in front of the embedded Python store.
//
// Every exported function is expr_res fn(int nargs, expr_res *args). The
// simulator rejects structure and array arguments, so each argument is one
// integer and the result is one integer. 0 is not a valid token id.

#include <Python.h>

#include <act/actsim_ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <string>

namespace {

PyObject *g_api = nullptr;

std::wstring widen(const char *utf8) {
  std::size_t n = std::mbstowcs(nullptr, utf8, 0);
  if (n == static_cast<std::size_t>(-1)) {
    return {};
  }
  std::wstring out(n, L'\0');
  std::mbstowcs(&out[0], utf8, n);
  return out;
}

bool init_python() {
  if (Py_IsInitialized()) {
    return true;
  }
  const char *home = std::getenv("QWEN_PYTHON_HOME");
  if (home == nullptr || home[0] == '\0') {
    std::fprintf(stderr, "qwen bridge: QWEN_PYTHON_HOME is not set\n");
    return false;
  }
  std::wstring home_w = widen(home);
  if (home_w.empty()) {
    std::fprintf(stderr, "qwen bridge: could not widen QWEN_PYTHON_HOME\n");
    return false;
  }

  PyConfig config;
  PyConfig_InitIsolatedConfig(&config);
  config.install_signal_handlers = 0;
  config.site_import = 1;
  PyStatus status = PyConfig_SetString(&config, &config.home, home_w.c_str());
  if (PyStatus_Exception(status)) {
    std::fprintf(stderr, "qwen bridge: PyConfig_SetString: %s\n",
                 status.err_msg ? status.err_msg : "unknown");
    PyConfig_Clear(&config);
    return false;
  }
  status = Py_InitializeFromConfig(&config);
  if (PyStatus_Exception(status)) {
    std::fprintf(stderr, "qwen bridge: Py_InitializeFromConfig: %s\n",
                 status.err_msg ? status.err_msg : "unknown");
    PyConfig_Clear(&config);
    return false;
  }
  PyConfig_Clear(&config);
  return true;
}

bool insert_path(const char *dir) {
  PyObject *sys = PyImport_ImportModule("sys");
  if (sys == nullptr) {
    PyErr_Print();
    return false;
  }
  PyObject *path = PyObject_GetAttrString(sys, "path");
  Py_DECREF(sys);
  if (path == nullptr) {
    PyErr_Print();
    return false;
  }
  PyObject *item = PyUnicode_FromString(dir);
  int rc = PyList_Insert(path, 0, item);
  Py_DECREF(item);
  Py_DECREF(path);
  if (rc != 0) {
    PyErr_Print();
    return false;
  }
  return true;
}

bool ensure_runtime() {
  if (g_api != nullptr) {
    return true;
  }
  if (!init_python()) {
    return false;
  }
  const char *site = std::getenv("QWEN_PYTHON_PATH");
  const char *root = std::getenv("QWEN_RUNTIME");
  if (site == nullptr || root == nullptr) {
    std::fprintf(stderr,
                 "qwen bridge: QWEN_PYTHON_PATH and QWEN_RUNTIME must be set\n");
    return false;
  }
  if (!insert_path(site) || !insert_path(root)) {
    return false;
  }
  g_api = PyImport_ImportModule("runtime.bridge_api");
  if (g_api == nullptr) {
    PyErr_Print();
    return false;
  }
  return true;
}

expr_res out_of(unsigned long value) {
  expr_res out;
  out.v = value;
  out.width = 32;
  return out;
}

unsigned long py_call(const char *name, const unsigned long *args, int nargs) {
  if (!ensure_runtime()) {
    return 0;
  }
  PyObject *fn = PyObject_GetAttrString(g_api, name);
  if (fn == nullptr) {
    PyErr_Print();
    return 0;
  }
  PyObject *tuple = PyTuple_New(nargs);
  for (int i = 0; i < nargs; i++) {
    PyTuple_SET_ITEM(tuple, i, PyLong_FromUnsignedLong(args[i]));
  }
  PyObject *ret = PyObject_CallObject(fn, tuple);
  Py_DECREF(tuple);
  Py_DECREF(fn);
  if (ret == nullptr) {
    PyErr_Print();
    return 0;
  }
  unsigned long value = PyLong_AsUnsignedLong(ret);
  Py_DECREF(ret);
  if (PyErr_Occurred()) {
    PyErr_Print();
    return 0;
  }
  return value;
}

expr_res call_checked(const char *name, int expect, int argc, expr_res *args) {
  if (argc != expect || (expect > 0 && args == nullptr)) {
    std::fprintf(stderr, "qwen bridge: %s expects %d argument%s\n", name, expect,
                 expect == 1 ? "" : "s");
    return out_of(0);
  }
  unsigned long buf[8];
  for (int i = 0; i < argc; i++) {
    buf[i] = args[i].v;
  }
  return out_of(py_call(name, buf, argc));
}

}  // namespace

// Component ids are layer * 16 + op (config/stage_map.json).

extern "C" expr_res qwen_f_init(int argc, expr_res *args) {
  return call_checked("f_init", 0, argc, args);
}

extern "C" expr_res qwen_f_source_prompt(int argc, expr_res *args) {
  return call_checked("f_source_prompt", 1, argc, args);
}

extern "C" expr_res qwen_f_source_layer(int argc, expr_res *args) {
  return call_checked("f_source_layer", 1, argc, args);
}

extern "C" expr_res qwen_f_fork(int argc, expr_res *args) {
  return call_checked("f_fork", 2, argc, args);
}

extern "C" expr_res qwen_f_compute1(int argc, expr_res *args) {
  return call_checked("f_compute1", 2, argc, args);
}

extern "C" expr_res qwen_f_compute2(int argc, expr_res *args) {
  return call_checked("f_compute2", 3, argc, args);
}

extern "C" expr_res qwen_f_commit(int argc, expr_res *args) {
  return call_checked("f_commit", 2, argc, args);
}

extern "C" expr_res qwen_f_finish_layer(int argc, expr_res *args) {
  return call_checked("f_finish_layer", 1, argc, args);
}

extern "C" expr_res qwen_f_continue_after(int argc, expr_res *args) {
  return call_checked("f_continue_after", 1, argc, args);
}

extern "C" expr_res qwen_f_decode_input(int argc, expr_res *args) {
  return call_checked("f_decode_input", 1, argc, args);
}

extern "C" expr_res qwen_f_finish_generate(int argc, expr_res *args) {
  return call_checked("f_finish_generate", 1, argc, args);
}
