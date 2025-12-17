//! Objective-C runtime bindings for Metal API access.
//! Provides low-level interop with Apple's Metal framework.

const std = @import("std");

// Objective-C runtime types
pub const id = ?*anyopaque;
pub const SEL = ?*anyopaque;
pub const Class = ?*anyopaque;
pub const IMP = *const fn () callconv(.c) void;

// Metal-specific types
pub const MTLSize = extern struct {
    width: u64,
    height: u64,
    depth: u64,

    pub fn make(w: u64, h: u64, d: u64) MTLSize {
        return .{ .width = w, .height = h, .depth = d };
    }

    pub fn make1D(w: u64) MTLSize {
        return make(w, 1, 1);
    }
};

pub const MTLOrigin = extern struct {
    x: u64,
    y: u64,
    z: u64,
};

pub const MTLRegion = extern struct {
    origin: MTLOrigin,
    size: MTLSize,
};

// MTLResourceOptions
pub const MTLResourceStorageModeShared: u64 = 0 << 4;
pub const MTLResourceStorageModeManaged: u64 = 1 << 4;
pub const MTLResourceStorageModePrivate: u64 = 2 << 4;
pub const MTLResourceCPUCacheModeDefaultCache: u64 = 0;
pub const MTLResourceCPUCacheModeWriteCombined: u64 = 1;
pub const MTLResourceHazardTrackingModeDefault: u64 = 0 << 8;
pub const MTLResourceHazardTrackingModeUntracked: u64 = 1 << 8;

// External Objective-C runtime functions
extern "c" fn objc_msgSend() void;
extern "c" fn objc_getClass(name: [*:0]const u8) Class;
extern "c" fn sel_registerName(name: [*:0]const u8) SEL;
extern "c" fn objc_retain(obj: id) id;
extern "c" fn objc_release(obj: id) void;
extern "c" fn objc_autorelease(obj: id) id;

// Metal framework function
extern "c" fn MTLCreateSystemDefaultDevice() id;

pub fn createSystemDefaultDevice() id {
    return MTLCreateSystemDefaultDevice();
}

/// Get Objective-C class by name.
pub fn getClass(name: [:0]const u8) Class {
    return objc_getClass(name.ptr);
}

/// Register/get selector by name.
pub fn sel(name: [:0]const u8) SEL {
    return sel_registerName(name.ptr);
}

/// Retain an Objective-C object.
pub fn retain(obj: id) id {
    return objc_retain(obj);
}

/// Release an Objective-C object.
pub fn release(obj: id) void {
    if (obj != null) {
        objc_release(obj);
    }
}

/// Autorelease an Objective-C object.
pub fn autorelease(obj: id) id {
    return objc_autorelease(obj);
}

/// Type-safe wrapper for objc_msgSend.
/// Handles the complex calling convention of Objective-C message sending.
pub fn msgSend(target: id, selector: SEL, args: anytype) id {
    const ArgsType = @TypeOf(args);
    const args_info = @typeInfo(ArgsType);

    if (args_info != .@"struct" or !args_info.@"struct".is_tuple) {
        @compileError("Expected tuple argument");
    }

    const fields = args_info.@"struct".fields;

    return switch (fields.len) {
        0 => @as(*const fn (id, SEL) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector),
        1 => @as(*const fn (id, SEL, @TypeOf(args[0])) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector, args[0]),
        2 => @as(*const fn (id, SEL, @TypeOf(args[0]), @TypeOf(args[1])) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector, args[0], args[1]),
        3 => @as(*const fn (id, SEL, @TypeOf(args[0]), @TypeOf(args[1]), @TypeOf(args[2])) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector, args[0], args[1], args[2]),
        4 => @as(*const fn (id, SEL, @TypeOf(args[0]), @TypeOf(args[1]), @TypeOf(args[2]), @TypeOf(args[3])) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector, args[0], args[1], args[2], args[3]),
        5 => @as(*const fn (id, SEL, @TypeOf(args[0]), @TypeOf(args[1]), @TypeOf(args[2]), @TypeOf(args[3]), @TypeOf(args[4])) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector, args[0], args[1], args[2], args[3], args[4]),
        6 => @as(*const fn (id, SEL, @TypeOf(args[0]), @TypeOf(args[1]), @TypeOf(args[2]), @TypeOf(args[3]), @TypeOf(args[4]), @TypeOf(args[5])) callconv(.c) id, @ptrCast(&objc_msgSend))(target, selector, args[0], args[1], args[2], args[3], args[4], args[5]),
        else => @compileError("Too many arguments for msgSend"),
    };
}

/// Send message with void return type.
pub fn msgSendVoid(target: id, selector: SEL, args: anytype) void {
    _ = msgSend(target, selector, args);
}

/// Send message expecting boolean return.
pub fn msgSendBool(target: id, selector: SEL, args: anytype) bool {
    const result = msgSend(target, selector, args);
    return @as(usize, @intFromPtr(result)) != 0;
}

/// Send message expecting integer return.
pub fn msgSendInt(comptime T: type, target: id, selector: SEL, args: anytype) T {
    const result = msgSend(target, selector, args);
    return @intCast(@as(usize, @intFromPtr(result)));
}

/// Create NSString from Zig string slice.
pub fn createNSString(str: []const u8) id {
    const NSString = getClass("NSString");
    const alloc_sel = sel("alloc");
    const init_sel = sel("initWithBytes:length:encoding:");

    const allocated = msgSend(NSString, alloc_sel, .{});
    const NSUTF8StringEncoding: u64 = 4;

    return msgSend(allocated, init_sel, .{ str.ptr, @as(u64, str.len), NSUTF8StringEncoding });
}

/// Create NSNumber from integer.
pub fn createNSNumber(value: i64) id {
    const NSNumber = getClass("NSNumber");
    return msgSend(NSNumber, sel("numberWithLongLong:"), .{value});
}

/// Create NSNumber from float.
pub fn createNSNumberFloat(value: f64) id {
    const NSNumber = getClass("NSNumber");
    return msgSend(NSNumber, sel("numberWithDouble:"), .{value});
}

/// Create NSArray from slice of objects.
pub fn createNSArray(objects: []const id) id {
    const NSArray = getClass("NSArray");
    return msgSend(NSArray, sel("arrayWithObjects:count:"), .{ objects.ptr, @as(u64, objects.len) });
}

/// Create NSDictionary from arrays of keys and values.
pub fn createNSDictionary(keys: []const id, values: []const id) id {
    std.debug.assert(keys.len == values.len);
    const NSDictionary = getClass("NSDictionary");
    return msgSend(NSDictionary, sel("dictionaryWithObjects:forKeys:count:"), .{ values.ptr, keys.ptr, @as(u64, keys.len) });
}

/// Get string contents from NSString.
pub fn getNSStringContents(ns_string: id) ?[*:0]const u8 {
    if (ns_string == null) return null;
    return @ptrCast(msgSend(ns_string, sel("UTF8String"), .{}));
}

/// Dispatch queue types for Metal.
pub const DispatchQueue = struct {
    pub fn getMain() id {
        const dispatch_get_main_queue_fn = @extern(*const fn () callconv(.C) id, .{ .name = "dispatch_get_main_queue" });
        return dispatch_get_main_queue_fn();
    }

    pub fn getGlobal(qos: u64) id {
        const dispatch_get_global_queue_fn = @extern(*const fn (u64, u64) callconv(.C) id, .{ .name = "dispatch_get_global_queue" });
        return dispatch_get_global_queue_fn(qos, 0);
    }
};

/// Quality of service classes for dispatch queues.
pub const QOS_CLASS_USER_INTERACTIVE: u64 = 0x21;
pub const QOS_CLASS_USER_INITIATED: u64 = 0x19;
pub const QOS_CLASS_DEFAULT: u64 = 0x15;
pub const QOS_CLASS_UTILITY: u64 = 0x11;
pub const QOS_CLASS_BACKGROUND: u64 = 0x09;
