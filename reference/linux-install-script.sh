V=1.1.0
P=linux-aarch64
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-$P.tar.gz" &&
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-$P.tar.gz.sha256" &&
sha256sum -c "tensorfold-$V-$P.tar.gz.sha256" &&
tar -xzf "tensorfold-$V-$P.tar.gz" &&
cd "tensorfold-$V-$P" &&
bin/tensorfold-native --version &&
bin/tensorfold-native capabilities --json