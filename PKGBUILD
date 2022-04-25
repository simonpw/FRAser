# Maintainer: Simon Williams <simon@clockcycles.net>
pkgname=FRAser
pkgver=1.0
pkgrel=1
epoch=
pkgdesc="A command line frequency response analyser, written in Python"
arch=(any)
url="https://github.com/simonpw/FRAser"
license=('GPL')
groups=()
depends=('python>=3.0' 'python-sounddevice' 'python-soundfile' 'python-numpy' 'python-scipy' 'python-tabulate')
makedepends=('git')
checkdepends=()
optdepends=('python-matplotlib')
provides=()
conflicts=()
replaces=()
backup=()
options=()
install=
changelog=
source=("git+https://github.com/simonpw/FRAser.git")
noextract=()
md5sums=('SKIP')

package() {
        cd "$srcdir/$pkgname"
        install -Dm755 FRAser.py ${pkgdir}/usr/bin/fraser
        install -Dm644 COPYING ${pkgdir}/usr/share/licenses/fraser/COPYING
}
