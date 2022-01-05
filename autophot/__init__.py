from pkg_resources import get_distribution

try:
    __version__ = get_distribution('autophot').version
except:
    __version__ = 'Test'
