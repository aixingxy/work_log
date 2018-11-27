cd ~
mkdir ~/.pip

cat > ~/.pip/pip.conf << EOF

[global]

timeout = 6000

index-url = https://pypi.doubanio.com/simple/

[install]

use-mirrors = true

mirrors = https://pypi.doubanio.com/simple/

trusted-host = pypi.douban.com
EOF
