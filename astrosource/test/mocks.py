from unittest.mock import MagicMock

from astropy.table import Table, Column
import numpy as np

def create_table_cols():
    mag = Column(np.array([16.82 ,    np.nan, 16.287, 16.566, 17.006], dtype=np.float32))
    emag = Column(np.array([0.018,   np.nan, 0.009, 0.02 , 0.11 ], dtype=np.float32))
    return mag, emag

def create_coords():
    ra = np.array([154.74983, 154.90837, 154.98908, 155.03358, 155.13282])
    dec = np.array([-9.90814, -9.80628, -9.66889, -9.97419, -9.73022])
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
    q = {'II/336/apass9' : t }
    return q

def mock_vizier_query_region_apass_v(*args, **kwargs):
    t = mock_vizier_query_region()
    m,e = create_table_cols()
    t.add_column(m, name='Vmag')
    t.add_column(e, name='e_Vmag')
    q = {'II/336/apass9' : t }
    return q

def mock_vizier_query_region_ps_r(*args, **kwargs):
    t = mock_vizier_query_region()
    m,e = create_table_cols()
    qual = Column(np.array([52,52,52,3,52],dtype='uint8'))
    t.add_column(m, name='rmag')
    t.add_column(e, name='e_rmag')
    t.add_column(qual, name='Qual')
    q = {'II/349/ps1' : t }
    return q

def mock_vizier_query_region_sdss_r(*args, **kwargs):
    t = mock_vizier_query_region_sdss()
    m,e = create_table_cols()
    qual = Column(np.array([52,3,3,3,52],dtype='uint8'))
    t.add_column(m, name='rmag')
    t.add_column(e, name='e_rmag')
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
