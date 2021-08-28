mod utils;

use wasm_bindgen::prelude::*;

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use polynomial::Polynomial;
use std::f64::consts::PI;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

const TWO_PI: f64 = PI + PI;

#[wasm_bindgen(inspectable)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct CoordCelestial {
    pub ra: f64,  // radians
    pub dec: f64, // radians
}

#[wasm_bindgen]
impl CoordCelestial {
    #[wasm_bindgen(constructor)]
    pub fn new(ra: f64, dec: f64) -> Self {
        CoordCelestial { ra, dec }
    }

    pub fn to_xyz(&self) -> Point3dJs {
        Point3dJs::new(
            self.dec.cos() * self.ra.cos(),
            self.dec.cos() * self.ra.sin(),
            self.dec.sin(),
        )
    }
}

pub type CoordCartesian = Vector2<f64>;
pub type CoordUnitSphere = Vector3<f64>;

impl From<CoordUnitSphere> for CoordCelestial {
    fn from(spherical: CoordUnitSphere) -> Self {
        let ra: f64 = spherical.y.atan2(spherical.x);
        let ra: f64 = if ra < 0f64 { ra + TWO_PI } else { ra };
        let dec: f64 = spherical.z.asin();

        CoordCelestial { ra, dec }
    }
}

impl From<&CoordUnitSphere> for CoordCelestial {
    fn from(spherical: &CoordUnitSphere) -> Self {
        let ra: f64 = spherical.y.atan2(spherical.x);
        let ra: f64 = if ra < 0f64 { ra + TWO_PI } else { ra };
        let dec: f64 = spherical.z.asin();

        CoordCelestial { ra, dec }
    }
}

impl Into<CoordUnitSphere> for CoordCelestial {
    fn into(self) -> CoordUnitSphere {
        CoordUnitSphere::new(
            self.dec.cos() * self.ra.cos(),
            self.dec.cos() * self.ra.sin(),
            self.dec.sin(),
        )
    }
}

#[wasm_bindgen(inspectable)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Point2dJs {
    pub x: f64,
    pub y: f64,
}

#[wasm_bindgen]
impl Point2dJs {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64) -> Self {
        Point2dJs { x, y }
    }
}

#[wasm_bindgen(inspectable)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Point3dJs {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[wasm_bindgen]
impl Point3dJs {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3dJs { x, y, z }
    }

    pub fn to_celestial(&self) -> CoordCelestial {
        let ra: f64 = self.y.atan2(self.x);
        let ra: f64 = if ra < 0f64 { ra + TWO_PI } else { ra };
        let dec: f64 = self.z.asin();

        CoordCelestial { ra, dec }
    }
}

#[wasm_bindgen(inspectable)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Mtx2x2Js {
    m11: f64,
    m12: f64,
    m21: f64,
    m22: f64
}

#[wasm_bindgen]
impl Mtx2x2Js {
    #[wasm_bindgen(constructor)]
    pub fn new(m11: f64, m12: f64, m21: f64, m22: f64) -> Self {
        Mtx2x2Js { m11, m12, m21, m22 }
    }
}

impl From<Point2dJs> for Vector2<f64> {
    fn from(p: Point2dJs) -> Self {
        Vector2::new(p.x, p.y)
    }
}

impl From<Point3dJs> for Vector3<f64> {
    fn from(p: Point3dJs) -> Self {
        Vector3::new(p.x, p.y, p.z)
    }
}

impl From<Mtx2x2Js> for Matrix2<f64> {
    fn from(m: Mtx2x2Js) -> Self {
        Matrix2::new(m.m11, m.m12, m.m21, m.m22)
    }
}

impl From<Vector2<f64>> for Point2dJs {
    fn from(p: Vector2<f64>) -> Self {
        Point2dJs {
            x: p.x,
            y: p.y,
        }
    }
}

impl From<Vector3<f64>> for Point3dJs {
    fn from(p: Vector3<f64>) -> Self {
        Point3dJs {
            x: p.x,
            y: p.y,
            z: p.z,
        }
    }
}

#[wasm_bindgen(inspectable)]
#[derive(Debug, Clone)]
pub struct WcsTan {
    // fiducial point in pixel space
    crpix: Vector2<f64>,

    // fiducial point in world space
    crval: CoordCelestial,

    // Matrix for the linear transformation of relative pixel coordinates
    // (u,v) onto "intermediate world coordinates", which are in degrees
    // (x,y). The x,y coordinates are on the tangent plane.
    //
    //   u = pixel_x - crpix0
    //   v = pixel_y - crpix1
    //
    //   x  = [cd00 cd01] * u
    //   y    [cd10 cd11]   v
    //
    // where x,y are in intermediate world coordinates (i.e. x points
    // along negative ra and y points to positive dec) and u,v are in pixel
    // coordinates.
    cd: Matrix2<f64>,
    cd_inv: Matrix2<f64>,

    // spherical rotation matrices
    mtx_w2i: Matrix3<f64>,
    mtx_i2w: Matrix3<f64>,

    // SIP parameters
    sip_params: Option<(Vec<f64>, Vec<f64>)>,

    // cached polynomials
    sip_polynomial: Option<(Polynomial<f64>, Polynomial<f64>)>,
}

#[wasm_bindgen]
impl WcsTan {
    #[wasm_bindgen(constructor)]
    pub fn new(crpix: Point2dJs, crval: CoordCelestial, cd: Mtx2x2Js) -> WcsTan {

        let crpix: Vector2<f64> = crpix.into();
        let cd: Matrix2<f64> = cd.into();

        let mtx_w2i = WcsTan::create_world_to_iwc_matrix(crval);
        let mtx_i2w = mtx_w2i.transpose();
        let cd_inv: Matrix2<f64> = cd.try_inverse().unwrap();

        WcsTan {
            crpix,
            crval,
            cd,
            cd_inv,
            mtx_w2i,
            mtx_i2w,

            sip_params: None,
            sip_polynomial: None,
        }
    }

    pub fn get_crpix(&self) -> Point2dJs {
        Point2dJs {
            x: self.crpix.x,
            y: self.crpix.y,
        }
    }

    pub fn get_crval(&self) -> CoordCelestial {
        CoordCelestial {
            ra: self.crval.ra,
            dec: self.crval.dec,
        }
    }

    /*
    fn setup_polynomial(&mut self) {
        if let Some(sip_params) = &self.sip_params {
            self.sip_polynomial = Some((
                Polynomial::new(sip_params.0.clone()),
                Polynomial::new(sip_params.0.clone()),
            ))
        }
    }
    */

    pub fn with_updated_sip(&self, new_sip_params_x: Vec<f64>, new_sip_params_y: Vec<f64>) -> WcsTan {
        let updated_sip_params = if let Some((x_params, y_params)) = &self.sip_params {
            (
                new_sip_params_x
                    .iter()
                    .zip(x_params.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
                new_sip_params_y
                    .iter()
                    .zip(y_params.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
            )
        } else {
            (new_sip_params_x, new_sip_params_y)
        };

        //debug!("SIP params updated to {:?}", &updated_sip_params);

        let updated_sip_polynomial = (
            Polynomial::new(updated_sip_params.0.clone()),
            Polynomial::new(updated_sip_params.1.clone()),
        );

        WcsTan {
            crpix: self.crpix,
            crval: self.crval,
            cd: self.cd,
            cd_inv: self.cd_inv,
            mtx_w2i: self.mtx_w2i,
            mtx_i2w: self.mtx_i2w,

            sip_params: Some(updated_sip_params),
            sip_polynomial: Some(updated_sip_polynomial),
        }
    }

    pub fn with_blank_sip(&self) -> WcsTan {
        WcsTan {
            crpix: self.crpix,
            crval: self.crval,
            cd: self.cd,
            cd_inv: self.cd_inv,
            mtx_w2i: self.mtx_w2i,
            mtx_i2w: self.mtx_i2w,

            sip_params: None,
            sip_polynomial: None,
        }
    }

    fn create_world_to_iwc_matrix(crval: CoordCelestial) -> Matrix3<f64> {
        let (sin_ra_p, cos_ra_p) = crval.ra.sin_cos();
        let (sin_dec_p, cos_dec_p) = crval.dec.sin_cos();

        Matrix3::new(
            cos_ra_p * sin_dec_p,
            sin_ra_p * sin_dec_p,
            -cos_dec_p,
            -sin_ra_p,
            cos_ra_p,
            0f64,
            cos_ra_p * cos_dec_p,
            sin_ra_p * cos_dec_p,
            sin_dec_p,
        )
    }

    fn world_2_iwc(&self, world: &Vector3<f64>) -> Vector3<f64> {
        self.mtx_w2i * world
    }

    fn iwc_2_world(&self, iwc: &Vector3<f64>) -> Vector3<f64> {
        self.mtx_i2w * iwc
    }

    // TODO: consider tan_project as it may be faster and we spend about 8%
    //       of cpu time in here
    fn iwc_2_proj_plane(iwc: &Vector3<f64>) -> Option<Vector2<f64>> {
        if iwc.z <= 0f64 {
            return None;
        }

        let w = (1f64 - iwc.z * iwc.z).sqrt();
        // getting rid of the to_degrees here gives us similar results to tan_project
        let rt = (w / iwc.z).to_degrees();

        let phi_mtx = iwc.y.atan2(iwc.x);

        Some(Vector2::new(rt * phi_mtx.sin(), -rt * phi_mtx.cos()))
    }

    fn proj_plane_2_iwc(&self, proj: &Vector2<f64>) -> Vector3<f64> {
        let radius = proj.magnitude();

        let phi = proj.x.atan2(-proj.y);

        let theta = (1f64 / radius.to_radians()).atan();

        (CoordCelestial {
            ra: phi,
            dec: theta,
        })
        .into()
    }

    fn proj_plane_2_pix(&self, proj: &Vector2<f64>) -> Vector2<f64> {
        let pix_coords: Vector2<f64> = self.cd_inv * proj;

        if let Some((p0, p1)) = &self.sip_polynomial {
            pix_coords + self.crpix + Vector2::new(p0.eval(pix_coords.x), p1.eval(pix_coords.y))
        } else {
            pix_coords + self.crpix
        }
    }

    fn proj_plane_2_pix_no_sip(&self, proj: &Vector2<f64>) -> Vector2<f64> {
        (self.cd_inv * proj) + self.crpix
    }

    fn pix_2_proj_plane(&self, pix: &Vector2<f64>) -> Vector2<f64> {
        self.cd * (pix - self.crpix)

        // TODO: this SIP inversion is wrong
        /*
        if let Some((p0, p1)) = &self.sip_polynomial {
            self.cd * (pix - self.crpix - Vector2::new(p0.eval(pix.x), p1.eval(pix.y)))
        } else {
            self.cd * (pix - self.crpix)
        }*/
    }

    pub fn world_2_pix(&self, world: Point3dJs) -> Option<Point2dJs> {
        let world: Vector3<f64> = world.into();

        let iwc = self.world_2_iwc(&world);
        let proj_plane = WcsTan::iwc_2_proj_plane(&iwc);
        proj_plane.map(|pp| self.proj_plane_2_pix(&pp)).map(|p| p.into())
    }

    pub fn world_2_pix_no_sip(&self, world: Point3dJs) -> Option<Point2dJs> {
        let world: Vector3<f64> = world.into();

        let iwc = self.world_2_iwc(&world);
        let proj_plane = WcsTan::iwc_2_proj_plane(&iwc);
        proj_plane.map(|pp| self.proj_plane_2_pix_no_sip(&pp)).map(|p| p.into())
    }

    pub fn pix_2_world(&self, pix: Point2dJs) -> Point3dJs {
        let pix: Vector2<f64> = pix.into();

        let proj_plane = self.pix_2_proj_plane(&pix);
        let iwc = self.proj_plane_2_iwc(&proj_plane);
        self.iwc_2_world(&iwc).into()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;

    const PROJ_EPSILON: f64 = 0.000000001;

    #[test]
    fn test_world_2_iwc_2_world() {
        let crpix = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);
        let crval = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_radec = CoordCelestial {
            ra: 193.50728997f64.to_radians(),
            dec: 55.95982296f64.to_radians(),
        };

        let alioth_xyz = alioth_radec.to_xyz().into();

        let expected_alioth_iwc_xyz = Point3dJs::new(
            0.0007526231030527963,
            0.056576195545634325,
            0.9983980006270279,
        );

        let iwc_xyz = wcs.world_2_iwc(&alioth_xyz);

        assert_relative_eq!(iwc_xyz.x, expected_alioth_iwc_xyz.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(iwc_xyz.y, expected_alioth_iwc_xyz.y, epsilon = PROJ_EPSILON);
        assert_relative_eq!(iwc_xyz.z, expected_alioth_iwc_xyz.z, epsilon = PROJ_EPSILON);

        let back_to_world_xyz = wcs.iwc_2_world(&iwc_xyz);

        assert_relative_eq!(back_to_world_xyz.x, alioth_xyz.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(back_to_world_xyz.y, alioth_xyz.y, epsilon = PROJ_EPSILON);
        assert_relative_eq!(back_to_world_xyz.z, alioth_xyz.z, epsilon = PROJ_EPSILON);

        let back_to_radec: CoordCelestial = back_to_world_xyz.into();

        assert_relative_eq!(back_to_radec.ra, alioth_radec.ra, epsilon = PROJ_EPSILON);
        assert_relative_eq!(back_to_radec.dec, alioth_radec.dec, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_iwc_2_proj_plane_2_iwc() {
        let crpix = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);
        let crval = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_iwc = Vector3::new(
            0.0007526231030527963,
            0.056576195545634325,
            0.9983980006270279,
        );

        let expected_alioth_proj = Point2dJs::new(3.2467785628936436f64, -0.0431913198362728f64);

        let proj_plane = WcsTan::iwc_2_proj_plane(&alioth_iwc).unwrap();

        assert_relative_eq!(proj_plane.x, expected_alioth_proj.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(proj_plane.y, expected_alioth_proj.y, epsilon = PROJ_EPSILON);

        let back_to_iwc = wcs.proj_plane_2_iwc(&proj_plane);

        assert_relative_eq!(back_to_iwc.x, alioth_iwc.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(back_to_iwc.y, alioth_iwc.y, epsilon = PROJ_EPSILON);
        assert_relative_eq!(back_to_iwc.z, alioth_iwc.z, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_proj_plane_2_pix_2_proj_plane() {
        let crpix = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);
        let crval = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_proj = Vector2::new(3.2467785628936436f64, -0.0431913198362728f64);

        let expected_alioth_pix = Vector2::new(1927.0413397194534f64, 2115.3157651975043f64);

        let result = wcs.proj_plane_2_pix(&alioth_proj);

        assert_relative_eq!(result.x, expected_alioth_pix.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.y, expected_alioth_pix.y, epsilon = PROJ_EPSILON);

        let back_to_proj = wcs.pix_2_proj_plane(&result);

        assert_relative_eq!(back_to_proj.x, alioth_proj.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(back_to_proj.y, alioth_proj.y, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_pix_2_proj_plane() {
        let crpix = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);
        let crval = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_pix = Vector2::new(1927.0413397194534f64, 2115.3157651975043f64);

        let expected_alioth_proj = Vector2::new(3.2467785628936436f64, -0.0431913198362728f64);

        let result = wcs.pix_2_proj_plane(&alioth_pix);

        assert_relative_eq!(result.x, expected_alioth_proj.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.y, expected_alioth_proj.y, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_world_2_pix() {
        let crpix = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);
        let crval = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_xyz = Point3dJs::new(-0.5442908710222142, -0.1307459229751705, 0.828645250603206);

        let expected_alioth_pix = Vector2::new(1927.0413397194534f64, 2115.3157651975043f64);

        let result = wcs.world_2_pix(alioth_xyz).unwrap();

        assert_relative_eq!(result.x, expected_alioth_pix.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.y, expected_alioth_pix.y, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_pix_2_world() {
        let crpix = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);
        let crval = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_pix = Point2dJs::new(1927.0413397194534f64, 2115.3157651975043f64);

        let expected_alioth_xyz =
            Vector3::new(-0.5442908710222142, -0.1307459229751705, 0.828645250603206);

        let result = wcs.pix_2_world(alioth_pix);

        assert_relative_eq!(result.x, expected_alioth_xyz.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.y, expected_alioth_xyz.y, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.z, expected_alioth_xyz.z, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_world_2_pix_2_world() {
        let crpix: Point2dJs = Point2dJs::new(2172.5994830347236f64, 2218.8150445202364f64);

        let crval: CoordCelestial = CoordCelestial {
            ra: 3.2760967777267447f64,
            dec: 0.9798160384441142f64,
        };
        let cd: Mtx2x2Js = Mtx2x2Js::new(
            -0.011164520045582788f64,
            -0.0048815774641172f64,
            0.0048815774641172f64,
            -0.011164520045582788f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        let alioth_radec = CoordCelestial {
            ra: 193.50728997f64.to_radians(),
            dec: 55.95982296f64.to_radians(),
        };

        let alioth_xyz: Point3dJs = alioth_radec.to_xyz();

        let expected_alioth_pix = Point2dJs::new(1927.0413397194534f64, 2115.3157651975043f64);

        let result = wcs.world_2_pix(alioth_xyz).unwrap();

        assert_relative_eq!(result.x, expected_alioth_pix.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.y, expected_alioth_pix.y, epsilon = PROJ_EPSILON);

        let result = wcs.pix_2_world(result);

        assert_relative_eq!(result.x, alioth_xyz.x, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.y, alioth_xyz.y, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.z, alioth_xyz.z, epsilon = PROJ_EPSILON);

        let result = result.to_celestial();

        assert_relative_eq!(result.ra, alioth_radec.ra, epsilon = PROJ_EPSILON);
        assert_relative_eq!(result.dec, alioth_radec.dec, epsilon = PROJ_EPSILON);
    }

    #[test]
    fn test_world_2_pix_two() {
        // From SAM_4222.JPG

        let crpix = Point2dJs::new(3181.286553668395f64, 2015.3096655692111f64);
        let crval = CoordCelestial {
            ra: 2.89606535f64,
            dec: 1.077758845f64,
        };
        let cd = Mtx2x2Js::new(
            -0.012648362747604055f64,
            -0.001115440909685504f64,
            0.0011483136431246982f64,
            -0.012661946361956078f64,
        );

        let wcs = WcsTan::new(crpix, crval, cd);

        // DUBHE: [165.932, 61.751]
        // ALKAID: [206.8852, 49.3133]
        // MERAK: [165.4603, 56.3824]
        // ALIOTH: [193.5073, 55.9598]

        let dubhe_xyz = Point3dJs::new(-0.45910891449532026, 0.11504763400370943, 0.8808989990578245);
        let alkaid_xyz = Point3dJs::new(-0.5814589564761689, -0.2948021361094878, 0.7582856865845008);
        let merak_xyz = Point3dJs::new(-0.535916223361377, 0.1389936001803212, 0.8327512117355289);
        let alioth_xyz = Point3dJs::new(-0.5442908710222142, -0.1307459229751705, 0.828645250603206);

        let expected_dubhe_pix = Vector2::new(3181.286553668395f64, 2015.3096655692111f64);
        let expected_alkaid_pix = Vector2::new(1017.1925122920688f64, 2194.822679036195f64);
        let expected_merak_pix = Vector2::new(3164.682886334279f64, 2438.9729863301413f64);
        let expected_alioth_pix = Vector2::new(1955.896866874826f64, 2114.7223628042207f64);

        let result_dubhe = wcs.world_2_pix(dubhe_xyz).unwrap();
        let result_alkaid = wcs.world_2_pix(alkaid_xyz).unwrap();
        let result_merak = wcs.world_2_pix(merak_xyz).unwrap();
        let result_alioth = wcs.world_2_pix(alioth_xyz).unwrap();

        assert_relative_eq!(result_dubhe.x, expected_dubhe_pix.x, epsilon = 1f64);
        assert_relative_eq!(result_dubhe.y, expected_dubhe_pix.y, epsilon = 1f64);

        assert_relative_eq!(result_alkaid.x, expected_alkaid_pix.x, epsilon = 1f64);
        assert_relative_eq!(result_alkaid.y, expected_alkaid_pix.y, epsilon = 1f64);

        assert_relative_eq!(result_merak.x, expected_merak_pix.x, epsilon = 1f64);
        assert_relative_eq!(result_merak.y, expected_merak_pix.y, epsilon = 1f64);
        
        assert_relative_eq!(result_alioth.x, expected_alioth_pix.x, epsilon = 1f64);
        assert_relative_eq!(result_alioth.y, expected_alioth_pix.y, epsilon = 1f64);
    }
}