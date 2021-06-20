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
        let cd_inv: Matrix2<f64> = cd.try_inverse().unwrap().transpose();

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

    pub fn with_updated_sip(&self, new_sip_params: (Vec<f64>, Vec<f64>)) -> WcsTan {
        let updated_sip_params = if let Some((x_params, y_params)) = &self.sip_params {
            (
                new_sip_params
                    .0
                    .iter()
                    .zip(x_params.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
                new_sip_params
                    .1
                    .iter()
                    .zip(y_params.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
            )
        } else {
            new_sip_params
        };

        debug!("SIP params updated to {:?}", &updated_sip_params);

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
    }*/

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
        // TODO: should this invert SIP?
        self.cd * (pix - self.crpix)
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
