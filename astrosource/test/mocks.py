from unittest.mock import MagicMock

from astropy.table import Table, Column
import numpy as np

def create_table_cols():
    mag = Column(np.array([12.33699989,12.03499985,11.31900024,12.20699978,11.48400021,11.88599968,12.09300041], dtype=np.float32))
    emag = Column(np.array([0.025,0.021,0.025,0.035,0.022,0.027,0.015], dtype=np.float32))
    return mag, emag

def create_coords():
    ra = np.array([163.096971,163.1466597,163.159242,163.197136,163.4236044,163.3569756,163.3740879])
    dec = np.array([-49.8792031,-49.8609692,-50.0239071,-49.8522255,-49.8430644,-49.9384119,-50.0038352])
    return ra, dec

def mock_vizier_query_region(*args, **kwargs):
    ra, dec = create_coords()
    return Table([ra, dec], names=('RAJ2000', 'DEJ2000'), dtype=('float64', 'float64'))

def mock_vizier_query_region_sdss(*args, **kwargs):
    ra, dec = create_coords()
    return Table([ra, dec], names=('RA_ICRS', 'DE_ICRS'), dtype=('float64', 'float64'))


def mock_vizier_query_region_vsx(*args, **kwargs):
    t = mock_vizier_query_region()
    q = {'B/vsx/vsx' : t }
    return q

def mock_vizier_query_region_apass_b(*args, **kwargs):
    t = mock_vizier_query_region()
    m,e = create_table_cols()
    t.add_column(m, name='Bmag')
    t.add_column(e, name='e_Bmag')
    t.add_column(m, name='Vmag')
    t.add_column(e, name='e_Vmag')
    q = {'II/336/apass9' : t }
    return q

def mock_vizier_query_region_apass_v(*args, **kwargs):
    t = mock_vizier_query_region()
    m,e = create_table_cols()
    t.add_column(m, name='Vmag')
    t.add_column(e, name='e_Vmag')
    t.add_column(m, name='Bmag')
    t.add_column(e, name='e_Bmag')
    q = {'II/336/apass9' : t }
    return q

def mock_vizier_query_region_ps_r(*args, **kwargs):
    t = mock_vizier_query_region()
    m,e = create_table_cols()
    qual = Column(np.array([52,52,52,3,52,3,52],dtype='uint8'))
    t.add_column(m, name='rmag')
    t.add_column(e, name='e_rmag')
    t.add_column(m, name='imag')
    t.add_column(e, name='e_imag')
    t.add_column(qual, name='Qual')
    q = {'II/349/ps1' : t }
    return q

def mock_vizier_query_region_sdss_r(*args, **kwargs):
    t = mock_vizier_query_region_sdss()
    m,e = create_table_cols()
    qual = Column(np.array([52,3,3,3,52,3,52],dtype='uint8'))
    t.add_column(m, name='rmag')
    t.add_column(e, name='e_rmag')
    t.add_column(m, name='imag')
    t.add_column(e, name='e_imag')
    t.add_column(qual, name='Q')
    q = {'V/147/sdss12' : t }
    return q

def mock_vizier_apass_b(*args, **kwargs):
    mock = MagicMock(query_region=mock_vizier_query_region_apass_b)
    return mock

def mock_vizier_apass_v(*args, **kwargs):
    mock = MagicMock(query_region=mock_vizier_query_region_apass_v)
    return mock

def mock_vizier_ps_r(*args, **kwargs):
    mock = MagicMock(query_region=mock_vizier_query_region_ps_r)
    return mock

def mock_vizier_sdss_r(*args, **kwargs):
    mock = MagicMock(query_region=mock_vizier_query_region_sdss_r)
    return mock
