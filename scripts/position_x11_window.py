#!/usr/bin/env python3
"""Move one named X11 client window without desktop automation packages."""

import argparse
import ctypes
import time


Display = ctypes.c_void_p
Window = ctypes.c_ulong


class ClientMessageData(ctypes.Union):
    _fields_ = [
        ('bytes', ctypes.c_char * 20),
        ('shorts', ctypes.c_short * 10),
        ('longs', ctypes.c_long * 5),
    ]


class XClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_int),
        ('serial', ctypes.c_ulong),
        ('send_event', ctypes.c_int),
        ('display', Display),
        ('window', Window),
        ('message_type', ctypes.c_ulong),
        ('format', ctypes.c_int),
        ('data', ClientMessageData),
    ]


class XEvent(ctypes.Union):
    _fields_ = [
        ('type', ctypes.c_int),
        ('xclient', XClientMessageEvent),
        ('padding', ctypes.c_long * 24),
    ]


def _load_x11():
    x11 = ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = Display
    x11.XCloseDisplay.argtypes = [Display]
    x11.XDefaultRootWindow.argtypes = [Display]
    x11.XDefaultRootWindow.restype = Window
    x11.XDefaultScreen.argtypes = [Display]
    x11.XDefaultScreen.restype = ctypes.c_int
    x11.XQueryTree.argtypes = [
        Display,
        Window,
        ctypes.POINTER(Window),
        ctypes.POINTER(Window),
        ctypes.POINTER(ctypes.POINTER(Window)),
        ctypes.POINTER(ctypes.c_uint),
    ]
    x11.XQueryTree.restype = ctypes.c_int
    x11.XFetchName.argtypes = [Display, Window, ctypes.POINTER(ctypes.c_char_p)]
    x11.XFetchName.restype = ctypes.c_int
    x11.XGetWindowProperty.argtypes = [
        Display,
        Window,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    x11.XGetWindowProperty.restype = ctypes.c_int
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XInternAtom.argtypes = [Display, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XSendEvent.argtypes = [
        Display, Window, ctypes.c_int, ctypes.c_long, ctypes.POINTER(XEvent)]
    x11.XSendEvent.restype = ctypes.c_int
    x11.XRaiseWindow.argtypes = [Display, Window]
    x11.XRaiseWindow.restype = ctypes.c_int
    x11.XIconifyWindow.argtypes = [Display, Window, ctypes.c_int]
    x11.XIconifyWindow.restype = ctypes.c_int
    x11.XUnmapWindow.argtypes = [Display, Window]
    x11.XUnmapWindow.restype = ctypes.c_int
    x11.XFlush.argtypes = [Display]
    x11.XFlush.restype = ctypes.c_int
    return x11


def _window_name(x11, display, window):
    name = ctypes.c_char_p()
    if not x11.XFetchName(display, window, ctypes.byref(name)) or not name.value:
        return ''
    try:
        return name.value.decode('utf-8', errors='replace')
    finally:
        x11.XFree(name)


def _window_pid(x11, display, window):
    """Read EWMH ``_NET_WM_PID`` for deterministic multi-window placement."""
    property_atom = x11.XInternAtom(display, b'_NET_WM_PID', 0)
    actual_type = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    item_count = ctypes.c_ulong()
    bytes_after = ctypes.c_ulong()
    value = ctypes.POINTER(ctypes.c_ubyte)()
    status = x11.XGetWindowProperty(
        display,
        window,
        property_atom,
        0,
        1,
        0,
        0,
        ctypes.byref(actual_type),
        ctypes.byref(actual_format),
        ctypes.byref(item_count),
        ctypes.byref(bytes_after),
        ctypes.byref(value),
    )
    if status != 0 or not value:
        return None
    try:
        if actual_format.value != 32 or item_count.value < 1:
            return None
        return int(ctypes.cast(value, ctypes.POINTER(ctypes.c_ulong))[0])
    finally:
        x11.XFree(value)


def _find_window(x11, display, root, *, title=None, pid=None):
    stack = [(root, 0)]
    deepest_match = None
    while stack:
        window, depth = stack.pop()
        matches = (
            (title is None or _window_name(x11, display, window) == title)
            and (pid is None or _window_pid(x11, display, window) == pid)
        )
        if matches:
            if deepest_match is None or depth > deepest_match[0]:
                deepest_match = (depth, window)
        returned_root = Window()
        returned_parent = Window()
        children = ctypes.POINTER(Window)()
        child_count = ctypes.c_uint()
        if x11.XQueryTree(
            display,
            window,
            ctypes.byref(returned_root),
            ctypes.byref(returned_parent),
            ctypes.byref(children),
            ctypes.byref(child_count),
        ):
            try:
                stack.extend(
                    (children[index], depth + 1)
                    for index in range(child_count.value))
            finally:
                if children:
                    x11.XFree(children)
    return None if deepest_match is None else deepest_match[1]


def _request_window_geometry(x11, display, root, window, x, y, width, height):
    """Ask an EWMH window manager to place a reparented client."""
    event = XEvent()
    event.xclient.type = 33  # ClientMessage
    event.xclient.serial = 0
    event.xclient.send_event = 1
    event.xclient.display = display
    event.xclient.window = window
    event.xclient.message_type = x11.XInternAtom(
        display, b'_NET_MOVERESIZE_WINDOW', 0)
    event.xclient.format = 32
    # EWMH bits 8..11 select x, y, width, and height respectively.
    event.xclient.data.longs[0] = (
        (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11))
    event.xclient.data.longs[1] = x
    event.xclient.data.longs[2] = y
    event.xclient.data.longs[3] = width
    event.xclient.data.longs[4] = height
    mask = (1 << 20) | (1 << 19)  # SubstructureRedirect/NotifyMask
    return x11.XSendEvent(
        display, root, 0, mask, ctypes.byref(event))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title')
    parser.add_argument('--pid', type=int)
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--timeout-sec', type=float, default=12.0)
    args = parser.parse_args()
    if args.title is None and args.pid is None:
        parser.error('at least one of --title or --pid is required')
    if not (args.minimize or args.hide) and None in (
            args.x, args.y, args.width, args.height):
        parser.error(
            '--x, --y, --width, and --height are required unless '
            '--minimize or --hide is used')

    x11 = _load_x11()
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError('could not open DISPLAY')
    try:
        root = x11.XDefaultRootWindow(display)
        deadline = time.monotonic() + max(0.1, args.timeout_sec)
        while time.monotonic() < deadline:
            window = _find_window(
                x11, display, root, title=args.title, pid=args.pid)
            if window is not None:
                if args.hide:
                    x11.XUnmapWindow(display, window)
                    x11.XFlush(display)
                    return 0
                if args.minimize:
                    x11.XIconifyWindow(
                        display, window, x11.XDefaultScreen(display))
                    x11.XFlush(display)
                    return 0
                # Re-issue briefly while Mutter finishes mapping/reparenting.
                for _ in range(10):
                    _request_window_geometry(
                        x11,
                        display,
                        root,
                        window,
                        args.x,
                        args.y,
                        args.width,
                        args.height,
                    )
                    x11.XRaiseWindow(display, window)
                    x11.XFlush(display)
                    time.sleep(0.2)
                return 0
            time.sleep(0.1)
    finally:
        x11.XCloseDisplay(display)
    selector = ', '.join(filter(None, [
        f'title={args.title!r}' if args.title is not None else None,
        f'pid={args.pid}' if args.pid is not None else None,
    ]))
    raise RuntimeError(f'window did not appear: {selector}')


if __name__ == '__main__':
    raise SystemExit(main())
