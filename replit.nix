{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.opencl-headers
    pkgs.ocl-icd
    pkgs.postgresql
    pkgs.openssl
  ];
}
